import argparse
import io
import os

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import soundfile as sf
import torch

from dataset_utils import (
    DEFAULT_DATASET,
    load_any_dataset,
    make_train_validation_test_split,
)
from jyutping import (
    add_tones_to_jyutping,
    annotation_to_legacy_texts,
    tone_less_tokens_to_jyutping,
)


DEFAULT_MODEL_ID = "hon9kon9ize/wav2vec2bert-jyutping"


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


def decode_full_jyutping(jyutping_text: str, tone_text: str) -> str:
    try:
        return add_tones_to_jyutping(tone_less_tokens_to_jyutping(jyutping_text), tone_text)
    except ValueError:
        return ""


def read_audio(audio) -> tuple[np.ndarray, int]:
    if audio.get("array") is not None:
        return np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]

    if audio.get("bytes") is not None:
        array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    else:
        array, sampling_rate = sf.read(audio["path"], dtype="float32")

    if array.ndim > 1:
        array = array.mean(axis=1)

    return array.astype(np.float32), sampling_rate


def evaluate(
    model_id: str,
    dataset: str,
    split: str,
    jyutping_column: str,
    tone_column: str,
    annotation_type: str,
    batch_size: int,
    max_samples: int,
    seed: int,
):
    from datasets import Audio
    from transformers import (
        SeamlessM4TFeatureExtractor,
        Wav2Vec2BertProcessor,
        Wav2Vec2CTCTokenizer,
    )

    from model import Wav2Vec2BertForCantonese

    dataset_dict = make_train_validation_test_split(
        load_any_dataset(dataset),
        test_size=500,
        validation_size=500,
        seed=seed,
    )
    eval_dataset = dataset_dict[split]
    if max_samples:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    eval_dataset = eval_dataset.cast_column("audio", Audio(decode=False))

    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    tone_tokenizer = Wav2Vec2CTCTokenizer(
        "tone_vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    patch_added_tokens(tokenizer)
    patch_added_tokens(tone_tokenizer)

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    model = Wav2Vec2BertForCantonese.from_pretrained(
        model_id,
        attention_dropout=0.2,
        hidden_dropout=0.2,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        add_adapter=True,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.config.update(
        {
            "vocab_size": len(tokenizer),
            "tone_vocab_size": len(tone_tokenizer),
        }
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    jyutping_predictions = []
    jyutping_references = []
    tone_predictions = []
    tone_references = []
    full_predictions = []
    full_references = []

    for start in range(0, len(eval_dataset), batch_size):
        rows = eval_dataset.select(range(start, min(start + batch_size, len(eval_dataset))))
        decoded_audio = [read_audio(row["audio"]) for row in rows]
        audio_arrays = [audio_array for audio_array, _ in decoded_audio]
        sampling_rates = [sampling_rate for _, sampling_rate in decoded_audio]
        sampling_rate = sampling_rates[0]
        if any(rate != sampling_rate for rate in sampling_rates):
            raise ValueError("Mixed sampling rates in one batch are not supported")

        inputs = processor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        input_features = inputs.input_features.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            outputs = model(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict=True,
            )

        jyutping_ids = torch.argmax(outputs.jyutping_logits, dim=-1).cpu()
        tone_ids = torch.argmax(outputs.tone_logits, dim=-1).cpu()
        batch_jyutping_predictions = processor.batch_decode(jyutping_ids)
        batch_tone_predictions = tone_tokenizer.batch_decode(tone_ids)

        for row, jyutping_prediction, tone_prediction in zip(
            rows, batch_jyutping_predictions, batch_tone_predictions
        ):
            tone_text = row.get(tone_column)
            jyutping_reference, tone_reference = annotation_to_legacy_texts(
                row[jyutping_column],
                tone_text=tone_text,
                annotation_type=annotation_type,
            )

            jyutping_predictions.append(jyutping_prediction)
            jyutping_references.append(jyutping_reference)
            tone_predictions.append(tone_prediction)
            tone_references.append(tone_reference)

            full_prediction = decode_full_jyutping(jyutping_prediction, tone_prediction)
            full_reference = decode_full_jyutping(jyutping_reference, tone_reference)
            if full_prediction and full_reference:
                full_predictions.append(full_prediction)
                full_references.append(full_reference)

    per = word_error_rate(jyutping_references, jyutping_predictions)
    ter = word_error_rate(tone_references, tone_predictions)
    ser = (
        word_error_rate(full_references, full_predictions)
        if full_references and full_predictions
        else None
    )

    print(f"model: {model_id}")
    print(f"dataset: {dataset}")
    print(f"split sizes: train={len(dataset_dict['train'])}, validation={len(dataset_dict['validation'])}, test={len(dataset_dict['test'])}")
    print(f"evaluated split: {split} ({len(eval_dataset)} samples)")
    print(f"PER: {per:.6f}")
    print(f"TER: {ter:.6f}")
    if ser is not None:
        print(f"SER: {ser:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["test", "validation", "train"], default="test")
    parser.add_argument("--jyutping_column", type=str, default="phone")
    parser.add_argument("--tone_column", type=str, default="tones")
    parser.add_argument(
        "--annotation_type",
        choices=["auto", "inline", "separate"],
        default="inline",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        args.model_id,
        args.dataset,
        args.split,
        args.jyutping_column,
        args.tone_column,
        args.annotation_type,
        args.batch_size,
        args.max_samples,
        args.seed,
    )
