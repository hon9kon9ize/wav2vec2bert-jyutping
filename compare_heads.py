"""Light head-design comparison on a small training subset.

Designs (identical data, base model, recipe and scoring; only heads/labels differ):

  single      one head over nucleus+tone tokens ("m aa4 | m aa1")          [our model]
  two_head    published design: onset/final pieces without tone ("m aa | m aa")
              plus a separate tone head ("4 1"), joined by the inference() logic
  single_aux  `single` plus an auxiliary tone-CTC loss; decoded from the main head only

    python compare_heads.py single --output_dir runs/single
    python compare_heads.py two_head --output_dir runs/two_head
    python compare_heads.py single_aux --output_dir runs/single_aux

Every design is scored the same way: both streams are turned into full toned jyutping
syllables, only the real (un-padded) CTC frames are decoded, and the syllable error
rate (SER) is the edit distance over syllables. Predictions are written to
<output_dir>/test_predictions.jsonl for later analysis.
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np

from ctc_utils import ctc_ids_to_token_text, ctc_output_length, load_ctc_tokenizer
from dataset_utils import (
    DEFAULT_DATASET,
    load_any_dataset,
    make_train_validation_test_split,
)
from jyutping import (
    ONSETS,
    SPECIAL_TOKENS,
    jyutping_to_nucleus_tone_label_tokens,
    normalize_jyutping_text,
    nucleus_tone_text_to_jyutping,
    split_nucleus_tone_syllable,
)

DESIGNS = ("single", "two_head", "single_aux")
LOGIT_KEYS = {
    "single": ("logits",),
    "two_head": ("jyutping_logits", "tone_logits"),
    "single_aux": ("logits", "tone_logits"),
}
LABEL_KEYS = {
    "single": ("labels",),
    "two_head": ("jyutping_labels", "tone_labels"),
    "single_aux": ("labels", "tone_labels"),
}


# --------------------------------------------------------------------------- labels
def piece_tokens(toned_text: str) -> list:
    """Published-style stream: onset / final pieces WITHOUT tone, "|" between syllables."""
    tokens = []
    for syllable in normalize_jyutping_text(toned_text).split():
        if tokens:
            tokens.append("|")
        parts = split_nucleus_tone_syllable(syllable)
        parts[-1] = parts[-1][:-1]  # drop the tone digit from the final piece
        tokens.extend(parts)
    return tokens


def pipe_between_tokens(tokens: list) -> list:
    """Old label style: "|" between EVERY token (what processor(text="m aa4 m aa1") produced)."""
    plain = [t for t in tokens if t != "|"]
    out = []
    for i, t in enumerate(plain):
        if i:
            out.append("|")
        out.append(t)
    return out


def tone_tokens(toned_text: str) -> list:
    return [s[-1] for s in normalize_jyutping_text(toned_text).split()]


def write_pieces_vocab(nucleus_tone_vocab_path: str, out_path: str) -> None:
    """Tone-less piece vocab derived from the SAME vocab the single head uses."""
    with open(nucleus_tone_vocab_path, encoding="utf-8") as f:
        source = json.load(f)
    pieces = set()
    for token in source:
        if token in SPECIAL_TOKENS:
            continue
        pieces.add(token if token in ONSETS else token[:-1])
    vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS + sorted(pieces))}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- decoding
def combine_streams(pieces: list, tones: list):
    """Exactly the logic of Wav2Vec2BertForCantonese.inference(): rebuild syllables from
    pieces, then zip tones on, repeating the last tone / truncating on a length mismatch.
    Returns (syllables, length_mismatch)."""
    joined = "".join(("_" + p) if p in ONSETS else (p + "_") for p in pieces)
    syllables = [s for s in re.sub(r"\s+", " ", joined.replace("_", " ")).strip().split(" ") if s]
    mismatch = len(tones) != len(syllables)
    if len(tones) > len(syllables):
        tones = tones[: len(syllables)]
    elif len(tones) < len(syllables):
        tones = tones + [tones[-1] if tones else ""] * (len(syllables) - len(tones))
    return [f"{s}{t}" for s, t in zip(syllables, tones)], mismatch


def row_to_syllables(design, tok_main, tok_tone, main_ids, tone_ids, length, collapse):
    main_text = ctc_ids_to_token_text(tok_main, main_ids, length, collapse)
    if design == "two_head":
        tones = ctc_ids_to_token_text(tok_tone, tone_ids, length, collapse).split()
        return combine_streams(main_text.split(), tones)
    return nucleus_tone_text_to_jyutping(main_text).split(), False


def edit_distance(ref, hyp) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def decode_all(design, predictions, label_ids, tok_main, tok_tone):
    """-> list of dicts {gold, pred, tone_len_mismatch}"""
    arrays = list(predictions)
    lengths = arrays.pop()  # LengthAware prediction_step appends the real CTC lengths
    main_pred = arrays[0].argmax(-1)
    tone_pred = arrays[1].argmax(-1) if len(arrays) > 1 else None
    labels = label_ids if isinstance(label_ids, (tuple, list)) else (label_ids,)
    rows = []
    for i, n in enumerate(lengths):
        pred, mismatch = row_to_syllables(
            design, tok_main, tok_tone, main_pred[i], None if tone_pred is None else tone_pred[i], int(n), True
        )
        gold, _ = row_to_syllables(
            design, tok_main, tok_tone, labels[0][i], labels[1][i] if len(labels) > 1 else None, None, False
        )
        rows.append({"gold": gold, "pred": pred, "tone_len_mismatch": bool(mismatch)})
    return rows


def score(rows, prefix=""):
    errors = sum(edit_distance(r["gold"], r["pred"]) for r in rows)
    total = sum(len(r["gold"]) for r in rows)
    out = {f"{prefix}ser": errors / max(total, 1), f"{prefix}exact_clips": float(np.mean([r["gold"] == r["pred"] for r in rows]))}
    out[f"{prefix}tone_len_mismatch"] = float(np.mean([r["tone_len_mismatch"] for r in rows]))
    return out


# --------------------------------------------------------------------------- trainer
def build_trainer_class():
    import torch
    from transformers import Trainer

    from ctc_utils import HeadLRMixin

    class HeadCompareTrainer(HeadLRMixin, Trainer):
        """Returns every logit stream plus each clip's real CTC length from evaluation."""

        def __init__(self, *args, logit_keys, label_keys, head_learning_rate=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.logit_keys, self.label_keys = logit_keys, label_keys
            self.head_learning_rate = head_learning_rate

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    out = model(**inputs, return_dict=True)
                loss = out["loss"].mean().detach()
                lengths = self.accelerator.unwrap_model(model)._get_feat_extract_output_lengths(
                    inputs["attention_mask"].sum(-1)
                ).to(torch.long)
            if prediction_loss_only:
                return loss, None, None
            logits = tuple(out[k].detach() for k in self.logit_keys) + (lengths,)
            labels = tuple(inputs[k] for k in self.label_keys)
            return loss, logits, labels[0] if len(labels) == 1 else labels

    return HeadCompareTrainer


@dataclass
class HeadCompareCollator:
    processor: object
    main_key: str  # "labels" or "jyutping_labels"
    with_tone: bool

    def __call__(self, features):
        batch = self.processor.pad(
            [{"input_features": f["input_features"]} for f in features],
            padding=True,
            return_tensors="pt",
        )
        main = self.processor.pad(
            labels=[{"input_ids": f["labels"]} for f in features], padding=True, return_tensors="pt"
        )
        batch[self.main_key] = main["input_ids"].masked_fill(main.attention_mask.ne(1), -100)
        if self.with_tone:
            tone = self.processor.pad(
                labels=[{"input_ids": f["tone_labels"]} for f in features], padding=True, return_tensors="pt"
            )
            batch["tone_labels"] = tone["input_ids"].masked_fill(tone.attention_mask.ne(1), -100)
        return batch


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("design", choices=DESIGNS)
    ap.add_argument("--model_id", default="facebook/w2v-bert-2.0")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--train_clips", type=int, default=6000)
    ap.add_argument("--val_clips", type=int, default=200)
    ap.add_argument("--test_clips", type=int, default=500)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--aux_weight", type=float, default=0.3)
    ap.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--no_gradient_checkpointing", action="store_true")
    ap.add_argument("--num_proc", type=int, default=4)
    ap.add_argument("--dataloader_num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_model", action="store_true", help="Save the final weights (2.4 GB).")
    ap.add_argument("--cache_dir", default="checkpoints/headcmp/cache", help="Audio-feature cache, reused across designs and runs.")
    ap.add_argument(
        "--label_style",
        choices=["syllable_pipe", "token_pipe"],
        default="syllable_pipe",
        help='"|" only between syllables (new) or between every token (what processor(text=...) used to produce).',
    )
    ap.add_argument("--head_lr_mult", type=float, default=1.0, help="Learning-rate multiplier for the output head(s).")
    args = ap.parse_args()

    from transformers import (
        SeamlessM4TFeatureExtractor,
        TrainingArguments,
        Wav2Vec2BertConfig,
        Wav2Vec2BertProcessor,
    )

    from model import Wav2Vec2BertForCantonese, Wav2Vec2BertForJyutpingCTC
    from model_aux import Wav2Vec2BertForJyutpingCTCAuxTone

    design = args.design
    os.makedirs(args.output_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    # ---- tokenizers
    if design == "two_head":
        main_vocab = os.path.join(args.output_dir, "vocab_pieces.json")
        write_pieces_vocab(os.path.join(here, "vocab_nucleus_tone.json"), main_vocab)
    else:
        main_vocab = os.path.join(here, "vocab_nucleus_tone.json")
    tok_main = load_ctc_tokenizer(main_vocab)
    tok_tone = load_ctc_tokenizer(os.path.join(here, "tone_vocab.json"))
    with_tone = design != "single"
    print(f"design={design}  main vocab={len(tok_main)}  tone vocab={len(tok_tone)}", flush=True)

    # ---- data: identical subset for every design (fixed split seed 42).
    # Audio features do not depend on the design or the labels, so they are cached on disk
    # and reused by every run; labels are cheap text work and are rebuilt each time.
    from datasets import load_from_disk

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(args.model_id)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tok_main)
    config = Wav2Vec2BertConfig.from_pretrained(args.model_id, add_adapter=True)
    sizes = {"train": args.train_clips, "validation": args.val_clips, "test": args.test_clips}
    feat_root = os.path.join(args.cache_dir, f"feat_{sizes['train']}_{sizes['validation']}_{sizes['test']}")

    def prepare_features(batch):
        audio = batch["audio"]
        feats = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        return {"input_features": feats, "output_lengths": ctc_output_length(len(feats), config)}

    if not all(os.path.isdir(os.path.join(feat_root, k)) for k in sizes):
        splits = make_train_validation_test_split(load_any_dataset(args.dataset), seed=42)
        for k, n in sizes.items():
            part = splits[k].select(range(n))
            part = part.map(
                prepare_features,
                num_proc=args.num_proc,
                remove_columns=[c for c in part.column_names if c != "phone"],
                desc=f"features {k}",
            )
            part.save_to_disk(os.path.join(feat_root, k))
    else:
        print(f"using cached features in {feat_root}", flush=True)
    ds = {k: load_from_disk(os.path.join(feat_root, k)) for k in sizes}

    def prepare_labels(batch):
        main_tokens = piece_tokens(batch["phone"]) if design == "two_head" else jyutping_to_nucleus_tone_label_tokens(batch["phone"])
        if args.label_style == "token_pipe":
            main_tokens = pipe_between_tokens(main_tokens)
        labels = tok_main.convert_tokens_to_ids(main_tokens)
        out = {"labels": labels, "label_lengths": len(labels), "n_unk": labels.count(tok_main.unk_token_id)}
        if with_tone:
            tone = tok_tone.convert_tokens_to_ids(tone_tokens(batch["phone"]))
            out["tone_labels"] = tone
            out["n_unk"] += tone.count(tok_tone.unk_token_id)
        return out

    ds = {k: v.map(prepare_labels, remove_columns=["phone"], desc=f"labels {k}") for k, v in ds.items()}
    for k, v in ds.items():
        n_unk, n_bad = sum(v["n_unk"]), sum(1 for a, b in zip(v["label_lengths"], v["output_lengths"]) if not 0 < a <= b)
        print(f"  {k}: {len(v)} clips | UNK label tokens: {n_unk} | CTC-impossible clips: {n_bad}", flush=True)
    # only the train split drops CTC-impossible clips; val/test keep every clip, as in the eval scripts
    ds["train"] = ds["train"].filter(lambda e: 0 < e["label_lengths"] <= e["output_lengths"], num_proc=args.num_proc)

    # ---- model
    common = dict(
        attention_dropout=0.2, hidden_dropout=0.2, feat_proj_dropout=0.0, mask_time_prob=0.0,
        layerdrop=0.0, add_adapter=True, ctc_loss_reduction="mean", ctc_zero_infinity=True,
        pad_token_id=tok_main.pad_token_id, vocab_size=len(tok_main),
    )
    if design == "single":
        model = Wav2Vec2BertForJyutpingCTC.from_pretrained(args.model_id, **common)
    elif design == "two_head":
        model = Wav2Vec2BertForCantonese.from_pretrained(args.model_id, tone_vocab_size=len(tok_tone), **common)
    else:
        model = Wav2Vec2BertForJyutpingCTCAuxTone.from_pretrained(
            args.model_id, tone_vocab_size=len(tok_tone), aux_tone_weight=args.aux_weight, **common
        )
    n_params = {n: p.numel() for n, p in model.named_parameters() if "head" in n}
    print(f"  head parameters: {n_params}", flush=True)

    collator = HeadCompareCollator(processor, LABEL_KEYS[design][0], with_tone)

    def compute_metrics(pred):
        return score(decode_all(design, pred.predictions, pred.label_ids, tok_main, tok_tone))

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.005,
        max_grad_norm=1.0,
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        gradient_checkpointing=not args.no_gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=25,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        label_names=list(LABEL_KEYS[design]),
        run_name=f"headcmp-{design}",
    )
    Trainer = build_trainer_class()
    trainer = Trainer(
        model=model, args=targs, data_collator=collator, compute_metrics=compute_metrics,
        train_dataset=ds["train"], eval_dataset=ds["validation"], processing_class=processor,
        logit_keys=LOGIT_KEYS[design], label_keys=LABEL_KEYS[design],
        head_learning_rate=args.learning_rate * args.head_lr_mult if args.head_lr_mult != 1.0 else None,
    )
    t0 = time.time()
    result = trainer.train()
    train_seconds = time.time() - t0

    # ---- final scoring on the test split, with predictions saved
    out = {"design": design, "train_clips": len(ds["train"]), "steps": args.max_steps,
           "train_seconds": round(train_seconds), "final_train_loss": result.training_loss}
    for name in ("validation", "test"):
        pred = trainer.predict(ds[name], metric_key_prefix=name)
        rows = decode_all(design, pred.predictions, pred.label_ids, tok_main, tok_tone)
        out.update({k: round(v, 5) for k, v in score(rows, prefix=f"{name}_").items()})
        if name == "test":
            with open(os.path.join(args.output_dir, "test_predictions.jsonl"), "w", encoding="utf-8") as f:
                for i, r in enumerate(rows):
                    f.write(json.dumps({"name": f"test_{i}", **r}, ensure_ascii=False) + "\n")
    curve = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in h.items() if k in ("step", "eval_ser", "eval_loss", "loss")}
             for h in trainer.state.log_history if "eval_ser" in h or "loss" in h]
    out["curve"] = curve
    json.dump(out, open(os.path.join(args.output_dir, "result.json"), "w"), indent=1)
    print("RESULT " + json.dumps({k: v for k, v in out.items() if k != "curve"}), flush=True)
    if args.save_model:
        trainer.save_model(os.path.join(args.output_dir, "final"))


if __name__ == "__main__":
    main()
