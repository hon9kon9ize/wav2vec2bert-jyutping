import argparse
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np

from jyutping import annotation_to_nucleus_tone_text, nucleus_tone_text_to_jyutping
from dataset_utils import (
    DEFAULT_DATASET,
    load_any_dataset,
    make_train_validation_test_split,
)


def extract_f0_features(audio_array, sampling_rate: int) -> list[float]:
    import librosa

    f0, _, _ = librosa.pyin(
        audio_array.astype(np.float32),
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sampling_rate,
    )
    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    return f0.astype(np.float32).tolist()


def word_error_rate(references: list[str], predictions: list[str]) -> float:
    errors = 0
    total = 0

    for reference, prediction in zip(references, predictions):
        ref_words = reference.split()
        pred_words = prediction.split()
        total += len(ref_words)

        previous = list(range(len(pred_words) + 1))
        for i, ref_word in enumerate(ref_words, start=1):
            current = [i]
            for j, pred_word in enumerate(pred_words, start=1):
                current.append(
                    min(
                        previous[j] + 1,
                        current[j - 1] + 1,
                        previous[j - 1] + (ref_word != pred_word),
                    )
                )
            previous = current

        errors += previous[-1]

    return errors / total if total else 0.0


def patch_added_tokens(tokenizer) -> None:
    from transformers import AddedToken

    for key in tokenizer.get_vocab().keys():
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


def train(
    model_id: str,
    dataset: str,
    output_dir: str,
    vocab_path: str,
    jyutping_column: str,
    tone_column: str,
    annotation_type: str,
    use_f0: bool,
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
):
    from transformers import (
        SeamlessM4TFeatureExtractor,
        Trainer,
        TrainingArguments,
        Wav2Vec2BertProcessor,
        Wav2Vec2CTCTokenizer,
        Wav2Vec2BertConfig,
    )

    from data import Wav2Vec2BertSingleDataCollatorCTCWithPadding
    from model import Wav2Vec2BertForJyutpingCTC

    print(f"Loading dataset: {dataset}", flush=True)
    ds = make_train_validation_test_split(load_any_dataset(dataset))
    print(
        "Split sizes: "
        f"train={len(ds['train'])}, validation={len(ds['validation'])}, test={len(ds['test'])}",
        flush=True,
    )
    if max_train_samples:
        ds["train"] = ds["train"].select(
            range(min(max_train_samples, len(ds["train"])))
        )
    if max_eval_samples:
        ds["validation"] = ds["validation"].select(
            range(min(max_eval_samples, len(ds["validation"])))
        )
    print(
        "Active sizes: "
        f"train={len(ds['train'])}, validation={len(ds['validation'])}",
        flush=True,
    )

    print(f"Loading tokenizer: {vocab_path}", flush=True)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    patch_added_tokens(tokenizer)

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )
    model_config = Wav2Vec2BertConfig.from_pretrained(model_id, add_adapter=True)

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["input_lengths"] = len(batch["input_features"])
        batch["output_lengths"] = get_ctc_output_length(
            batch["input_lengths"], model_config
        )
        if use_f0:
            batch["f0_features"] = extract_f0_features(
                audio["array"], audio["sampling_rate"]
            )

        tone_text = batch.get(tone_column)
        label_text = annotation_to_nucleus_tone_text(
            batch[jyutping_column],
            tone_text=tone_text,
            annotation_type=annotation_type,
        )
        batch["labels"] = processor(text=label_text).input_ids
        batch["label_lengths"] = len(batch["labels"])

        return batch

    print(f"Preprocessing dataset with num_proc={num_proc}", flush=True)
    remove_columns = ds["train"].column_names
    ds = ds.map(
        prepare_dataset,
        num_proc=num_proc,
        remove_columns=remove_columns,
        desc="Preparing audio features and labels",
    )
    before_filter = {split: len(ds[split]) for split in ds}
    ds = ds.filter(
        lambda example: 0 < example["label_lengths"] <= example["output_lengths"],
        num_proc=num_proc,
        desc="Filtering CTC-impossible examples",
    )
    after_filter = {split: len(ds[split]) for split in ds}
    print(
        "Filtered CTC-impossible examples: "
        + ", ".join(
            f"{split}={before_filter[split] - after_filter[split]}"
            for split in before_filter
        ),
        flush=True,
    )
    print("Preprocessing complete", flush=True)

    print(f"Loading model: {model_id}", flush=True)
    model = Wav2Vec2BertForJyutpingCTC.from_pretrained(
        model_id,
        attention_dropout=0.2,
        hidden_dropout=0.2,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        add_adapter=True,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        use_f0=use_f0,
    )

    data_collator = Wav2Vec2BertSingleDataCollatorCTCWithPadding(
        processor=processor, padding=True
    )

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = np.where(
            pred.label_ids == -100, processor.tokenizer.pad_token_id, pred.label_ids
        )

        pred_token_text = processor.batch_decode(pred_ids)
        label_token_text = processor.batch_decode(label_ids, group_tokens=False)
        pred_jyutping = [
            nucleus_tone_text_to_jyutping(text) for text in pred_token_text
        ]
        label_jyutping = [
            nucleus_tone_text_to_jyutping(text) for text in label_token_text
        ]

        token_error = word_error_rate(label_token_text, pred_token_text)
        syllable_error = word_error_rate(label_jyutping, pred_jyutping)

        return {"per": token_error, "ser": syllable_error}

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
        num_train_epochs=10,
        fp16=precision == "fp16",
        bf16=precision == "bf16",
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        overwrite_output_dir=True,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=100,
        learning_rate=learning_rate,
        weight_decay=0.005,
        warmup_steps=warmup_steps,
        max_grad_norm=max_grad_norm,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="ser",
        greater_is_better=False,
        report_to=None if report_to == "none" else report_to,
        run_name="wav2vec2-yue-nucleus-tone" + time.strftime("%Y-%m-%d-%H-%M-%S"),
    )

    trainer = Trainer(
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
    parser.add_argument("dataset", type=str, nargs="?", default=DEFAULT_DATASET)
    parser.add_argument("--output_dir", type=str, default="checkpoints_nucleus_tone")
    parser.add_argument("--vocab_path", type=str, default="vocab_nucleus_tone.json")
    parser.add_argument("--jyutping_column", type=str, default="phone")
    parser.add_argument("--tone_column", type=str, default="tones")
    parser.add_argument(
        "--annotation_type",
        choices=["auto", "inline", "separate"],
        default="inline",
        help="inline means jyutping has tones; separate means jyutping plus tone columns.",
    )
    parser.add_argument(
        "--use_f0",
        action="store_true",
        help="Add extracted F0 as an auxiliary pitch feature for the single CTC head.",
    )
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
        help="Enable length-grouped sampling. This can be slow with some Transformers versions.",
    )
    parser.add_argument(
        "--no_group_by_length",
        action="store_true",
        help="Deprecated compatibility flag; length grouping is disabled by default.",
    )
    parser.add_argument(
        "--no_gradient_checkpointing",
        action="store_true",
        help="Disable gradient checkpointing for faster training when memory allows.",
    )
    args = parser.parse_args()

    train(
        args.model_id,
        args.dataset,
        args.output_dir,
        args.vocab_path,
        args.jyutping_column,
        args.tone_column,
        args.annotation_type,
        args.use_f0,
        args.num_proc,
        args.max_train_samples,
        args.max_eval_samples,
        args.per_device_train_batch_size,
        args.per_device_eval_batch_size,
        args.gradient_accumulation_steps,
        args.report_to,
        args.max_steps,
        args.group_by_length and not args.no_group_by_length,
        args.precision,
        args.dataloader_num_workers,
        not args.no_gradient_checkpointing,
        args.learning_rate,
        args.warmup_steps,
        args.max_grad_norm,
    )
