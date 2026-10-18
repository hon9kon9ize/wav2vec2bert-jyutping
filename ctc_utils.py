"""Decoding and Trainer helpers shared by the nucleus+tone CTC scripts."""

from itertools import groupby
from typing import Optional, Sequence

import torch
from transformers import Trainer


def load_ctc_tokenizer(vocab_path: str):
    """Wav2Vec2CTCTokenizer whose multi-character vocab entries are kept whole."""
    from transformers import AddedToken, Wav2Vec2CTCTokenizer

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    for key, idx in tokenizer.get_vocab().items():
        if key not in tokenizer.special_tokens_map.values():
            tokenizer._added_tokens_decoder[idx] = AddedToken(
                key, lstrip=False, rstrip=False
            )
    return tokenizer


def ctc_output_length(input_length: int, config) -> int:
    """Number of CTC frames the encoder produces for `input_length` feature frames."""
    if not getattr(config, "add_adapter", False):
        return input_length
    padding = config.adapter_kernel_size // 2
    for _ in range(config.num_adapter_layers):
        input_length = (
            input_length + 2 * padding - config.adapter_kernel_size
        ) // config.adapter_stride + 1
    return input_length


def ctc_ids_to_token_text(
    tokenizer,
    ids: Sequence[int],
    length: Optional[int] = None,
    collapse: bool = True,
) -> str:
    """Turn frame-level CTC ids into space-separated pieces, e.g. "m aa4 m aa1".

    - ``length``: only the first ``length`` frames count. A batch is padded to its
      longest clip and the model still emits ids for the padding frames, so
      decoding all of them adds spurious tokens to the shorter clips.
    - ``collapse``: greedy CTC collapse (merge repeats, drop blanks). Pass False
      for reference labels, where repeats are real. Negative ids (-100 label
      padding) are dropped either way.
    - The word delimiter "|" is dropped, so a model trained with "|" between every
      token and one trained with "|" only between syllables give the same text
      (``tokenizer.decode`` would give "m aa4 m aa1" vs "maa4 maa1").
    """
    ids = [int(i) for i in (ids if length is None else ids[:length])]
    if collapse:
        ids = [key for key, _ in groupby(ids)]
    blank = tokenizer.pad_token_id
    ids = [i for i in ids if i >= 0 and i != blank]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return " ".join(t for t in tokens if t != tokenizer.word_delimiter_token)


class HeadLRMixin:
    """Train the freshly initialised output head(s) at their own learning rate.

    A uniform learning rate leaves single-head CTC fine-tuning stuck on the
    all-blank plateau at small batch sizes (loss ~3, SER ~100%). The same run with
    the head at 1e-3 and the encoder at 5e-5 reaches ~19% SER in 300 steps.

    Set ``head_learning_rate`` (absolute) on the trainer; None keeps one learning
    rate for everything. Head parameters are the ones whose name contains "head"
    (lm_head, jyutping_head, tone_head), with no weight decay.
    """

    head_learning_rate: Optional[float] = None

    def create_optimizer(self):
        if self.optimizer is not None or self.head_learning_rate is None:
            return super().create_optimizer()
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        head = [p for n, p in named if "head" in n]
        rest = [p for n, p in named if "head" not in n]
        optimizer_cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(
            [
                {
                    "params": rest,
                    "lr": self.args.learning_rate,
                    "weight_decay": self.args.weight_decay,
                },
                {"params": head, "lr": self.head_learning_rate, "weight_decay": 0.0},
            ],
            **kwargs,
        )
        return self.optimizer


class LengthAwareTrainer(HeadLRMixin, Trainer):
    """Trainer whose predictions are ``(logits, output_lengths)``.

    ``output_lengths`` is each clip's real number of CTC frames, so
    ``compute_metrics`` can ignore padding frames: ``pred.predictions`` is
    ``(logits, lengths)`` instead of just logits.
    """

    def __init__(self, *args, head_learning_rate: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.head_learning_rate = head_learning_rate

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        loss, logits, labels = super().prediction_step(
            model, inputs, prediction_loss_only, ignore_keys=ignore_keys
        )
        if logits is None:
            return loss, logits, labels

        unwrapped = self.accelerator.unwrap_model(model)
        with torch.no_grad():
            lengths = unwrapped._get_feat_extract_output_lengths(
                inputs["attention_mask"].sum(-1)
            ).to(device=logits.device, dtype=torch.long)
        return loss, (logits, lengths), labels


MERGER_PARTNER_TONES = {"2": ["5"], "5": ["2"], "3": ["6"], "6": ["3", "4"], "4": ["6"]}


def merger_partner_ids(tokenizer, vocab_size=None):
    """LongTensor (vocab_size, 2): for every rime+tone token the ids of the same rime with its merger-partner tone
    (2<->5, 3<->6, 4<->6; Mok & Wong 2010), padded with -1."""
    import torch

    vocab = tokenizer.get_vocab()
    n = vocab_size or len(tokenizer)
    out = torch.full((n, 2), -1, dtype=torch.long)
    for tok, i in vocab.items():
        if i < n and len(tok) > 1 and tok[-1] in MERGER_PARTNER_TONES and tok[:-1].isalpha():
            ids = [vocab[tok[:-1] + t] for t in MERGER_PARTNER_TONES[tok[-1]] if tok[:-1] + t in vocab]
            for k, j in enumerate(ids[:2]):
                out[i, k] = j
    return out
