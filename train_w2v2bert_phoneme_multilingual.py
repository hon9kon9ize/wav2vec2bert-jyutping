import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import torch
from pathlib import Path

from phoneme_utils import phonemes_to_tokens
from dataset_utils import make_train_validation_test_split

DEFAULT_DATASET_BASE = (
    "/home/pj24001684/ku40000295/jc/projects/phoneme_asr_dataset/output/hf"
)
DEFAULT_LANGS = ["yue", "zh", "en"]
LANG_IDS = {"yue": 0, "zh": 1, "en": 2}


def load_multilingual_dataset(base_path: str, langs: list):
    from datasets import load_from_disk, concatenate_datasets, DatasetDict

    parts = []
    for lang in langs:
        ds = load_from_disk(str(Path(base_path) / lang))
        if isinstance(ds, DatasetDict):
            # flatten all available splits into one dataset
            flat = concatenate_datasets([ds[s] for s in ds])
        else:
            flat = ds
        parts.append(flat)
        print(f"  {lang}: {len(flat)} examples", flush=True)

    return concatenate_datasets(parts)


def word_error_rate(references: list, predictions: list) -> float:
    errors = 0
    total = 0
    for ref, pred in zip(references, predictions):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        total += len(ref_tokens)
        prev = list(range(len(pred_tokens) + 1))
        for i, r in enumerate(ref_tokens, 1):
            cur = [i]
            for j, p in enumerate(pred_tokens, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != p)))
            prev = cur
        errors += prev[-1]
    return errors / total if total else 0.0


def patch_added_tokens(tokenizer) -> None:
    from transformers import AddedToken

    for key in tokenizer.get_vocab():
        if key not in tokenizer.special_tokens_map.values():
            idx = tokenizer.get_vocab()[key]
            tokenizer._added_tokens_decoder[idx] = AddedToken(
                key, lstrip=False, rstrip=False
            )


def get_ctc_output_length(input_length: int, config) -> int:
    if not getattr(config, "add_adapter", False):
        return input_length
    output_length = input_length
    padding = config.adapter_kernel_size // 2
    for _ in range(config.num_adapter_layers):
        output_length = (
            output_length + 2 * padding - config.adapter_kernel_size
        ) // config.adapter_stride + 1
    return output_length


def upsample_by_language(dataset, temperature: float, seed: int):
    """
    Temperature-based upsampling to counter language imbalance.

    Target proportion for language L is p_L ∝ n_L^temperature, achieved by
    duplicating minority-language examples only (the majority is the anchor and
    is never touched). The per-language repeat factor is:

        r_L = (n_max / n_L) ** (1 - temperature)

    temperature=1.0 -> r_L = 1 for all languages (no-op, natural distribution)
    temperature=0.5 -> r_L = sqrt(n_max / n_L) (moderate, the usual default)
    temperature=0.0 -> r_L = n_max / n_L (fully balanced, aggressive)

    Fractional factors duplicate the whole set floor(r_L) times, then add a
    random subset for the remainder. Expects an integer "lang_id" column.
    """
    from datasets import concatenate_datasets

    lang_ids = dataset["lang_id"]
    groups: dict = {}
    for i, lid in enumerate(lang_ids):
        groups.setdefault(lid, []).append(i)

    counts = {lid: len(idx) for lid, idx in groups.items()}
    n_max = max(counts.values())

    parts = []
    factors = {}
    for lid, idx in groups.items():
        factor = (n_max / counts[lid]) ** (1.0 - temperature)
        factors[lid] = factor
        subset = dataset.select(idx)
        whole = int(factor)
        for _ in range(whole):
            parts.append(subset)
        remainder = int(round((factor - whole) * len(idx)))
        if remainder > 0:
            parts.append(subset.shuffle(seed=seed).select(range(remainder)))

    upsampled = concatenate_datasets(parts).shuffle(seed=seed)
    return upsampled, counts, factors


def train(
    model_id: str,
    dataset_base: str,
    langs: list,
    output_dir: str,
    vocab_path: str,
    num_proc: int,
    max_train_samples: int,
    max_eval_samples: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    gradient_accumulation_steps: int,
    report_to: str,
    max_steps: int,
    group_by_length: bool,
    precision: str,
    dataloader_num_workers: int,
    gradient_checkpointing: bool,
    learning_rate: float,
    warmup_steps: int,
    max_grad_norm: float,
    add_adapter: bool,
    max_duration: float,
    lang_upsample_temperature: float,
    seed: int,
    dropout: float,
    no_lang_mask: bool,
    head_learning_rate: float = None,
):
    from transformers import (
        SeamlessM4TFeatureExtractor,
        TrainingArguments,
        Wav2Vec2BertProcessor,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2BertConfig,
    )
    from ctc_utils import LengthAwareTrainer
    from data import Wav2Vec2BertSingleDataCollatorCTCWithPadding
    from model import Wav2Vec2BertForJyutpingCTC

    print(f"Loading multilingual datasets from {dataset_base} — langs: {langs}", flush=True)
    raw = load_multilingual_dataset(dataset_base, langs)
    ds = make_train_validation_test_split(raw, seed=seed)
    print(
        f"Split: train={len(ds['train'])}, validation={len(ds['validation'])}, "
        f"test={len(ds['test'])}",
        flush=True,
    )

    if max_train_samples:
        ds["train"] = ds["train"].select(range(min(max_train_samples, len(ds["train"]))))
    if max_eval_samples:
        ds["validation"] = ds["validation"].select(
            range(min(max_eval_samples, len(ds["validation"])))
        )
    print(
        f"Active: train={len(ds['train'])}, validation={len(ds['validation'])}",
        flush=True,
    )

    if max_duration > 0:
        before_dur = {s: len(ds[s]) for s in ds}
        ds = ds.filter(
            lambda ex: len(ex["audio"]["array"]) / ex["audio"]["sampling_rate"] <= max_duration,
            num_proc=num_proc,
            desc=f"Filtering audio > {max_duration}s",
        )
        after_dur = {s: len(ds[s]) for s in ds}
        print(
            f"Duration filter (>{max_duration}s): "
            + ", ".join(f"{s}={before_dur[s] - after_dur[s]}" for s in before_dur),
            flush=True,
        )

    print(f"Loading tokenizer from {vocab_path}", flush=True)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    patch_added_tokens(tokenizer)

    vocab = tokenizer.get_vocab()
    specials = {tokenizer.pad_token, tokenizer.unk_token, tokenizer.word_delimiter_token}
    lang_masks = torch.zeros(len(LANG_IDS), len(vocab), dtype=torch.bool)
    for token, idx in vocab.items():
        if token in specials:
            lang_masks[:, idx] = True
        elif token.startswith("yue_"):
            lang_masks[LANG_IDS["yue"], idx] = True
        elif token.startswith("zh_"):
            lang_masks[LANG_IDS["zh"], idx] = True
        else:
            lang_masks[LANG_IDS["en"], idx] = True
    print(
        f"Lang masks: yue={lang_masks[0].sum()}, zh={lang_masks[1].sum()}, "
        f"en={lang_masks[2].sum()} active tokens",
        flush=True,
    )

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    model_config = Wav2Vec2BertConfig.from_pretrained(model_id, add_adapter=add_adapter)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["input_lengths"] = len(batch["input_features"])
        batch["output_lengths"] = get_ctc_output_length(
            batch["input_lengths"], model_config
        )

        token_list = phonemes_to_tokens(batch["phonemes"], batch["lang"])
        # Map tokens -> ids directly. Do NOT round-trip through
        # processor(text=" ".join(...)): the Wav2Vec2CTCTokenizer substitutes
        # every space with the word-delimiter "|", which inserts a spurious "|"
        # between onset and rime and triples the intended syllable separators,
        # producing a pipe-saturated target that breaks CTC alignment.
        batch["labels"] = tokenizer.convert_tokens_to_ids(token_list)
        batch["label_lengths"] = len(batch["labels"])
        batch["lang_id"] = LANG_IDS[batch["lang"]]
        return batch

    print(f"Preprocessing with num_proc={num_proc}", flush=True)
    remove_columns = ds["train"].column_names
    ds = ds.map(
        prepare_dataset,
        num_proc=num_proc,
        remove_columns=remove_columns,
        desc="Preparing features and labels",
    )

    before = {s: len(ds[s]) for s in ds}
    ds = ds.filter(
        lambda ex: 0 < ex["label_lengths"] <= ex["output_lengths"],
        num_proc=num_proc,
        desc="Filtering CTC-impossible examples",
    )
    after = {s: len(ds[s]) for s in ds}
    print(
        "Filtered: "
        + ", ".join(f"{s}={before[s] - after[s]}" for s in before),
        flush=True,
    )
    print("Preprocessing complete", flush=True)

    # Upsample low-resource languages in the TRAIN split only (val/test keep the
    # natural distribution for honest evaluation).
    if lang_upsample_temperature < 1.0:
        id_to_lang = {v: k for k, v in LANG_IDS.items()}
        ds["train"], counts, factors = upsample_by_language(
            ds["train"], lang_upsample_temperature, seed
        )
        print(
            f"Language upsampling (temperature={lang_upsample_temperature}):",
            flush=True,
        )
        for lid in sorted(counts):
            lang = id_to_lang.get(lid, str(lid))
            new_count = int(round(counts[lid] * factors[lid]))
            print(
                f"  {lang}: {counts[lid]} -> {new_count} (x{factors[lid]:.2f})",
                flush=True,
            )
        print(f"  train total: {len(ds['train'])}", flush=True)

    print(f"Loading model: {model_id}", flush=True)
    model = Wav2Vec2BertForJyutpingCTC.from_pretrained(
        model_id,
        attention_dropout=dropout,
        hidden_dropout=dropout,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        add_adapter=add_adapter,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    if not no_lang_mask:
        model.register_buffer("lang_masks", lang_masks)
    else:
        print("Language masking DISABLED (pure flat head)", flush=True)

    data_collator = Wav2Vec2BertSingleDataCollatorCTCWithPadding(
        processor=processor, padding=True
    )

    def compute_metrics(pred):
        # LengthAwareTrainer gives (logits, output_lengths); only the first
        # output_length frames of each clip are real, the rest is batch padding.
        logits, output_lengths = pred.predictions
        pred_ids = np.argmax(logits, axis=-1)
        label_ids = np.where(
            pred.label_ids == -100,
            processor.tokenizer.pad_token_id,
            pred.label_ids,
        )
        pred_strs = processor.batch_decode(
            [ids[:length] for ids, length in zip(pred_ids, output_lengths)]
        )
        label_strs = processor.batch_decode(label_ids, group_tokens=False)
        per = word_error_rate(label_strs, pred_strs)
        return {"per": per}

    training_args = TrainingArguments(
        output_dir=output_dir,
        label_names=["labels"],
        group_by_length=group_by_length,
        length_column_name="input_lengths",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_strategy="steps",
        max_steps=max_steps,
        num_train_epochs=4,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        overwrite_output_dir=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.005,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="per",
        greater_is_better=False,
        report_to=None if report_to == "none" else report_to,
        run_name="wav2vec2-multilingual-phoneme-" + time.strftime("%Y-%m-%d-%H-%M-%S"),
    )

    trainer = LengthAwareTrainer(
        head_learning_rate=head_learning_rate,
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=processor,
    )

    print("Starting training", flush=True)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str)
    parser.add_argument("dataset_base", type=str, nargs="?", default=DEFAULT_DATASET_BASE)
    parser.add_argument("--langs", nargs="+", default=DEFAULT_LANGS)
    parser.add_argument("--output_dir", type=str, default="checkpoints_multilingual_phoneme")
    parser.add_argument("--vocab_path", type=str, default="vocab_multilingual_phoneme.json")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=256)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--report_to", choices=["wandb", "none"], default="wandb")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument(
        "--precision",
        choices=["fp16", "bf16", "fp32"],
        default="fp16",
        help="RTX 3090 is usually fastest with fp16.",
    )
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--group_by_length",
        action="store_true",
        help="Enable length-grouped sampling.",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
    )
    parser.add_argument(
        "--no_adapter",
        action="store_true",
        help="Disable the conformer adapter layers (4x longer CTC output, fits more examples).",
    )
    parser.add_argument(
        "--max_duration",
        type=float,
        default=10.0,
        help="Drop audio clips longer than this many seconds (0 to disable).",
    )
    parser.add_argument(
        "--lang_upsample_temperature",
        type=float,
        default=0.5,
        help=(
            "Temperature for upsampling low-resource languages in the train "
            "split. 1.0 disables it (natural distribution); 0.5 uses "
            "sqrt(n_max/n_L) repeat factors; 0.0 fully balances languages."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--head_learning_rate",
        type=float,
        default=None,
        help=(
            "Separate learning rate for the output head. A uniform learning rate leaves "
            "single-head CTC on the all-blank plateau (loss ~3, PER ~1.0); on the Cantonese "
            "task --learning_rate 5e-5 --head_learning_rate 1e-3 fixed it."
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="attention/hidden dropout. Set 0.0 for overfit (memorization) tests.",
    )
    parser.add_argument(
        "--no_lang_mask",
        action="store_true",
        help="Disable per-language -inf logit masking (train a pure flat head).",
    )
    args = parser.parse_args()

    train(
        args.model_id,
        args.dataset_base,
        args.langs,
        args.output_dir,
        args.vocab_path,
        args.num_proc,
        args.max_train_samples,
        args.max_eval_samples,
        args.per_device_train_batch_size,
        args.per_device_eval_batch_size,
        args.gradient_accumulation_steps,
        args.report_to,
        args.max_steps,
        args.group_by_length,
        args.precision,
        args.dataloader_num_workers,
        not args.no_gradient_checkpointing,
        args.learning_rate,
        args.warmup_steps,
        args.max_grad_norm,
        not args.no_adapter,
        args.max_duration,
        args.lang_upsample_temperature,
        args.seed,
        args.dropout,
        args.no_lang_mask,
        args.head_learning_rate,
    )
