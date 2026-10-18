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


# --------------------------------------------------------------------------- sequential feature building
def stored_positions(n):
    """(perm, position) of the fixed shuffle (seed 42) that makes the train/validation/test split: shuffled position p is stored row
    perm[p], and stored row r sits at shuffled position position[r]. Checked against Dataset.shuffle by the caller."""
    perm = np.random.default_rng(42).permutation(n)
    position = np.empty(n, dtype=np.int64)
    position[perm] = np.arange(n)
    return perm, position


def select_features(dataset, want, phone_column, prepare, num_proc):
    """Features of the stored rows in `want` ({row: (kind, key)}), computed while the dataset is read in its stored order.

    `dataset.select(indices)` of the shuffled split is random access, which runs at ~5 clips/s on genkai's Lustre; reading all rows
    in order is fast and only the wanted ones are featurised, with the same `prepare` function as the normal path."""
    from datasets import Features, Sequence, Value

    def build(batch, rows):
        out = {"input_features": [], "output_lengths": [], "phone": [], "kind": [], "key": []}
        for j, row in enumerate(rows):
            if row not in want:
                continue
            done = prepare({"audio": batch["audio"][j]})
            out["input_features"].append(done["input_features"])
            out["output_lengths"].append(done["output_lengths"])
            out["phone"].append(batch[phone_column][j])
            out["kind"].append(want[row][0])
            out["key"].append(want[row][1])
        return out

    features = Features({"input_features": Sequence(Sequence(Value("float32"))), "output_lengths": Value("int64"),
                         "phone": Value("string"), "kind": Value("int8"), "key": Value("int64")})
    return dataset.map(build, batched=True, batch_size=64, with_indices=True, num_proc=num_proc,
                       remove_columns=dataset.column_names, features=features, desc="features of the wanted rows")


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
    ap.add_argument("--extra_train_dir", action="append", default=None,
                    help="directory with manifest.json (file, phone) and wavs, e.g. synth_tts.py output; its clips are added to the "
                         "training subset (repeatable). Features are cached under the cache dir by directory name.")
    ap.add_argument("--extra_train_limit", type=int, default=0,
                    help="use only the first N clips of every --extra_train_dir (after the voice filter)")
    ap.add_argument("--extra_train_ref_prefix", default=None,
                    help="keep only the extra training clips whose voice (manifest 'ref') starts with this prefix; use 'ref' to leave out "
                         "the voices of the labelled evaluation wavs, whose speaker would otherwise leak into training")
    ap.add_argument("--train_indices", default=None,
                    help="JSON list of indices into the fixed train split to train on instead of its first --train_clips clips "
                         "(make_relabel_experiment.py); their features are cached under the file's name")
    ap.add_argument("--extra_train_hf", default=None, metavar="PATH:COLUMN",
                    help="add rows of another Hugging Face dataset (its train split) to the training subset, e.g. dataset_emo_jyutping:labels; "
                         "audio is resampled to 16 kHz, features are cached under the cache dir by name")
    ap.add_argument("--extra_train_hf_indices", default=None, help="JSON list of the rows of --extra_train_hf to use (default: all)")
    ap.add_argument("--train_top_layers", type=int, default=0,
                    help="train only the top N encoder layers, the adapter and the head(s) and freeze everything below (the weights, gradients and "
                         "AdamW state of a full fine-tune do not fit an 11 GB MIG slice); 0 = train everything")
    ap.add_argument("--sequential_features", action="store_true",
                    help="with --train_indices / --extra_train_hf_indices: read the stored datasets in order and featurise only the wanted rows "
                         "(identical features; use it where random access is slow, e.g. genkai's Lustre)")
    ap.add_argument("--repeat_indices", default=None,
                    help="JSON list of train-split indices (needs --train_indices) whose clips are repeated --repeat_times times in training")
    ap.add_argument("--repeat_times", type=int, default=2, help="how often a --repeat_indices clip appears in the training data in total")
    ap.add_argument("--group_by_length", action="store_true", help="length-grouped batches (faster, less padding)")
    ap.add_argument("--phone_override", default=None,
                    help="JSON {train-split index: toned jyutping} that replaces the label of those clips (needs --train_indices)")
    ap.add_argument("--merger_smoothing", type=float, default=0.0,
                    help="single design only: weight of the merger-aware smoothing loss (keeps a share of the probability on the "
                         "merger-partner tones 2/5, 3/6, 4/6 at the frames where the model emits a token); 0 = off.")
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
    train_indices = json.load(open(args.train_indices)) if args.train_indices else None
    if args.phone_override and train_indices is None:
        raise SystemExit("--phone_override needs --train_indices")
    sizes = {"train": len(train_indices) if train_indices else args.train_clips, "validation": args.val_clips, "test": args.test_clips}
    train_tag = os.path.splitext(os.path.basename(args.train_indices))[0] + f"{len(train_indices)}" if train_indices else str(sizes["train"])
    feat_root = os.path.join(args.cache_dir, f"feat_{train_tag}_{sizes['validation']}_{sizes['test']}")

    def prepare_features(batch):
        audio = batch["audio"]
        feats = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        return {"input_features": feats, "output_lengths": ctc_output_length(len(feats), config)}

    if not all(os.path.isdir(os.path.join(feat_root, k)) for k in sizes) and args.sequential_features and train_indices is not None:
        from dataset_utils import flatten_dataset

        flat = flatten_dataset(load_any_dataset(args.dataset))
        perm, position = stored_positions(len(flat))
        phones = flat.select_columns(["phone"])
        shuffled = phones.shuffle(seed=42)
        for p in (0, 1, len(flat) // 3, len(flat) // 2, len(flat) - 1):
            assert shuffled[p] == phones[int(perm[p])], f"the permutation differs from Dataset.shuffle at {p}"
        want = {int(perm[src + 1000]): (1, int(src)) for src in train_indices}  # the first 1,000 shuffled rows are test and validation
        want.update({int(perm[q]): (3, q) for q in range(args.test_clips)})
        want.update({int(perm[500 + q]): (2, 500 + q) for q in range(args.val_clips)})
        everything = select_features(flat, want, "phone", prepare_features, args.num_proc)
        for kind, name in ((1, "train"), (2, "validation"), (3, "test")):
            part = everything.filter(lambda k: k == kind, input_columns=["kind"], num_proc=args.num_proc, desc=f"{name} rows")
            part = part.remove_columns(["kind"])
            part = part.rename_column("key", "src_idx") if kind == 1 else part.sort("key").remove_columns(["key"])
            part.save_to_disk(os.path.join(feat_root, name))
    elif not all(os.path.isdir(os.path.join(feat_root, k)) for k in sizes):
        splits = make_train_validation_test_split(load_any_dataset(args.dataset), seed=42)
        for k, n in sizes.items():
            part = splits[k].select(train_indices if (k == "train" and train_indices) else range(n))
            if k == "train" and train_indices:
                part = part.add_column("src_idx", list(train_indices))
            part = part.map(
                prepare_features,
                num_proc=args.num_proc,
                remove_columns=[c for c in part.column_names if c not in ("phone", "src_idx")],
                desc=f"features {k}",
            )
            part.save_to_disk(os.path.join(feat_root, k))
    else:
        print(f"using cached features in {feat_root}", flush=True)
    ds = {k: load_from_disk(os.path.join(feat_root, k)) for k in sizes}
    if args.phone_override:
        override = {int(k): v for k, v in json.load(open(args.phone_override)).items()}
        ds["train"] = ds["train"].map(lambda e: {"phone": override.get(e["src_idx"], e["phone"])}, desc="phone override")
        print(f"  {sum(i in override for i in ds['train']['src_idx'])} training labels overridden", flush=True)
    for extra in args.extra_train_dir or []:
        from datasets import Audio, Dataset, concatenate_datasets

        tag = os.path.basename(os.path.normpath(extra)) + (f"_{args.extra_train_ref_prefix}" if args.extra_train_ref_prefix else "") \
            + (f"_first{args.extra_train_limit}" if args.extra_train_limit else "")
        cache = os.path.join(args.cache_dir, f"extra_{tag}")
        if not os.path.isdir(cache):
            man = json.load(open(os.path.join(extra, "manifest.json")))
            if args.extra_train_ref_prefix:
                kept = [m for m in man if str(m.get("ref", "")).startswith(args.extra_train_ref_prefix)]
                print(f"  {extra}: keeping {len(kept)} of {len(man)} clips (voice prefix '{args.extra_train_ref_prefix}')", flush=True)
                man = kept
            if args.extra_train_limit:
                man = man[: args.extra_train_limit]
            part = Dataset.from_dict({"audio": [os.path.join(extra, m["file"]) for m in man], "phone": [m["phone"] for m in man]})
            part = part.cast_column("audio", Audio(sampling_rate=16_000))
            part = part.map(prepare_features, num_proc=args.num_proc, remove_columns=["audio"], desc=f"features {tag}")
            part.save_to_disk(cache)
        part = load_from_disk(cache)
        print(f"  extra training clips from {extra}: {len(part)}", flush=True)
        ds["train"] = concatenate_datasets([ds["train"], part])

    if args.extra_train_hf:
        from datasets import Audio, concatenate_datasets

        path, _, column = args.extra_train_hf.partition(":")
        rows = json.load(open(args.extra_train_hf_indices)) if args.extra_train_hf_indices else None
        tag = "hf_" + os.path.basename(os.path.normpath(path)) + (
            f"_{os.path.splitext(os.path.basename(args.extra_train_hf_indices))[0]}{len(rows)}" if rows else "")
        cache = os.path.join(args.cache_dir, f"extra_{tag}")
        if not os.path.isdir(cache):
            extra = load_any_dataset(path)
            extra = extra["train"] if hasattr(extra, "keys") else extra
            if args.sequential_features and rows is not None:
                extra = extra.select_columns(["audio", column]).cast_column("audio", Audio(sampling_rate=16_000))
                done = select_features(extra, {int(r): (1, int(r)) for r in rows}, column, prepare_features, args.num_proc)
                done.remove_columns(["kind", "key"]).save_to_disk(cache)
            else:
                if rows is not None:
                    extra = extra.select(rows)
                extra = extra.select_columns(["audio", column]).rename_column(column, "phone").cast_column("audio", Audio(sampling_rate=16_000))
                extra.map(prepare_features, num_proc=args.num_proc, remove_columns=["audio"], desc=f"features {tag}").save_to_disk(cache)
        part = load_from_disk(cache)
        if "src_idx" in ds["train"].column_names:
            part = part.add_column("src_idx", [-1] * len(part))
        print(f"  extra training clips from {path}: {len(part)}", flush=True)
        ds["train"] = concatenate_datasets([ds["train"], part])
    if args.repeat_indices:
        from datasets import concatenate_datasets

        again = set(json.load(open(args.repeat_indices)))
        repeated = ds["train"].filter(lambda i: i in again, input_columns=["src_idx"], num_proc=args.num_proc, desc="repeated clips")
        ds["train"] = concatenate_datasets([ds["train"]] + [repeated] * (args.repeat_times - 1))
        print(f"  {len(repeated)} clips appear {args.repeat_times} times: training clips {len(ds['train'])}", flush=True)

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
        if args.merger_smoothing > 0:
            from ctc_utils import merger_partner_ids

            pid = merger_partner_ids(tok_main, len(tok_main))
            model.set_merger_smoothing(args.merger_smoothing, pid)
            print(f"  merger-aware smoothing {args.merger_smoothing}: {int((pid[:, 0] >= 0).sum())} tokens have partner tones", flush=True)
    elif design == "two_head":
        model = Wav2Vec2BertForCantonese.from_pretrained(args.model_id, tone_vocab_size=len(tok_tone), **common)
    else:
        model = Wav2Vec2BertForJyutpingCTCAuxTone.from_pretrained(
            args.model_id, tone_vocab_size=len(tok_tone), aux_tone_weight=args.aux_weight, **common
        )
    if args.train_top_layers:
        cut = config.num_hidden_layers - args.train_top_layers

        def frozen(name):
            if name.startswith(("wav2vec2_bert.feature_projection", "wav2vec2_bert.masked_spec_embed")):
                return True
            layer = re.match(r"wav2vec2_bert\.encoder\.layers\.(\d+)\.", name)
            return bool(layer) and int(layer.group(1)) < cut

        for name, param in model.named_parameters():
            param.requires_grad = not frozen(name)
        print(f"  training the top {args.train_top_layers} of {config.num_hidden_layers} encoder layers + adapter + head: "
              f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.0f}M of "
              f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters", flush=True)
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
        # with frozen lower layers the checkpointed upper layers get inputs without grad: only the non-reentrant mode passes gradients on
        **({"gradient_checkpointing_kwargs": {"use_reentrant": False}} if args.train_top_layers else {}),
        group_by_length=args.group_by_length,
        length_column_name="output_lengths",
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
