import argparse
import io
import os
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import soundfile as sf
import torch

from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
from jyutping import (
    add_tones_to_jyutping,
    annotation_to_legacy_texts,
    annotation_to_nucleus_tone_text,
    nucleus_tone_text_to_jyutping,
    tone_less_tokens_to_jyutping,
)


DEFAULT_PUBLISHED_MODEL_ID = "hon9kon9ize/wav2vec2bert-jyutping"
DEFAULT_CHECKPOINT_MODEL_ID = "checkpoints/checkpoint-1524"
DEFAULT_BASE_MODEL_ID = "facebook/w2v-bert-2.0"


@dataclass
class Prediction:
    reference: str
    prediction: str
    token_reference: str = ""
    token_prediction: str = ""
    tone_reference: str = ""
    tone_prediction: str = ""
    strict_prediction: str | None = None
    length_mismatch: bool = False


def patch_added_tokens(tokenizer) -> None:
    from transformers import AddedToken

    for key in tokenizer.get_vocab().keys():
        if key not in tokenizer.special_tokens_map.values():
            idx = tokenizer.get_vocab()[key]
            tokenizer._added_tokens_decoder[idx] = AddedToken(
                key, lstrip=False, rstrip=False
            )


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


def split_base_tone(syllable: str) -> tuple[str, str]:
    if syllable and syllable[-1] in "123456":
        return syllable[:-1], syllable[-1]
    return syllable, ""


def edit_alignment(reference: Iterable[str], prediction: Iterable[str]):
    ref = list(reference)
    pred = list(prediction)
    dp = [[0] * (len(pred) + 1) for _ in range(len(ref) + 1)]

    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(pred) + 1):
        dp[0][j] = j

    for i, ref_token in enumerate(ref, start=1):
        for j, pred_token in enumerate(pred, start=1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (ref_token != pred_token),
            )

    ops = []
    i = len(ref)
    j = len(pred)
    while i or j:
        if (
            i
            and j
            and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != pred[j - 1])
        ):
            ops.append(("equal" if ref[i - 1] == pred[j - 1] else "sub", ref[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", pred[j - 1]))
            j -= 1

    return list(reversed(ops))


def classify_ops(reference: str, prediction: str) -> tuple[Counter, Counter]:
    counts = Counter()
    confusions = Counter()

    for op, ref_syllable, pred_syllable in edit_alignment(reference.split(), prediction.split()):
        if op == "equal":
            counts["correct"] += 1
            continue
        if op == "del":
            counts["deletion"] += 1
            confusions[(ref_syllable, "<del>")] += 1
            continue
        if op == "ins":
            counts["insertion"] += 1
            confusions[("<ins>", pred_syllable)] += 1
            continue

        ref_base, ref_tone = split_base_tone(ref_syllable)
        pred_base, pred_tone = split_base_tone(pred_syllable)
        if ref_base == pred_base and ref_tone != pred_tone:
            counts["tone_only"] += 1
        elif ref_base != pred_base and ref_tone == pred_tone:
            counts["base_only"] += 1
        else:
            counts["base_and_tone"] += 1
        confusions[(ref_syllable, pred_syllable)] += 1

    return counts, confusions


def word_error_rate(references: list[str], predictions: list[str]) -> float:
    total = 0
    errors = 0
    for reference, prediction in zip(references, predictions):
        ref_words = reference.split()
        total += len(ref_words)
        errors += sum(1 for op, _, _ in edit_alignment(ref_words, prediction.split()) if op != "equal")
    return errors / total if total else 0.0


def combine_base_and_tones(base_text: str, tone_text: str) -> tuple[str, bool]:
    base_syllables = tone_less_tokens_to_jyutping(base_text).split()
    tones = tone_text.split()
    if len(tones) == 1 and len(base_syllables) > 1 and len(tones[0]) > 1:
        tones = list(tones[0])

    full = []
    for index, syllable in enumerate(base_syllables):
        tone = tones[index] if index < len(tones) else "?"
        full.append(f"{syllable}{tone}")

    return " ".join(full), len(base_syllables) != len(tones)


def decode_full_jyutping(jyutping_text: str, tone_text: str) -> str | None:
    try:
        return add_tones_to_jyutping(tone_less_tokens_to_jyutping(jyutping_text), tone_text)
    except ValueError:
        return None


def load_eval_dataset(dataset: str, split: str, max_samples: int, seed: int):
    from datasets import Audio

    dataset_dict = make_train_validation_test_split(
        load_any_dataset(dataset),
        test_size=500,
        validation_size=500,
        seed=seed,
    )
    eval_dataset = dataset_dict[split]
    if max_samples:
        eval_dataset = eval_dataset.select(range(min(max_samples, len(eval_dataset))))
    return dataset_dict, eval_dataset.cast_column("audio", Audio(decode=False))


def predict_published(
    eval_dataset,
    model_id: str,
    jyutping_column: str,
    tone_column: str,
    annotation_type: str,
    batch_size: int,
) -> list[Prediction]:
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer

    from model import Wav2Vec2BertForCantonese

    tokenizer = Wav2Vec2CTCTokenizer(
        "vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    tone_tokenizer = Wav2Vec2CTCTokenizer(
        "tone_vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    patch_added_tokens(tokenizer)
    patch_added_tokens(tone_tokenizer)

    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_id)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)
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
    model.config.update({"vocab_size": len(tokenizer), "tone_vocab_size": len(tone_tokenizer)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions = []
    for start in range(0, len(eval_dataset), batch_size):
        rows = eval_dataset.select(range(start, min(start + batch_size, len(eval_dataset))))
        decoded_audio = [read_audio(row["audio"]) for row in rows]
        audio_arrays = [audio_array for audio_array, _ in decoded_audio]
        sampling_rates = [sampling_rate for _, sampling_rate in decoded_audio]
        sampling_rate = sampling_rates[0]

        inputs = processor(audio_arrays, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            outputs = model(
                input_features=inputs.input_features.to(device),
                attention_mask=attention_mask,
                return_dict=True,
            )

        jyutping_ids = torch.argmax(outputs.jyutping_logits, dim=-1)
        tone_ids = torch.argmax(outputs.tone_logits, dim=-1)
        if attention_mask is not None:
            output_lengths = model._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)
        else:
            output_lengths = torch.full(
                (jyutping_ids.shape[0],),
                jyutping_ids.shape[1],
                device=jyutping_ids.device,
                dtype=torch.long,
            )
        jyutping_predictions = [
            processor.decode(ids[:length].cpu())
            for ids, length in zip(jyutping_ids, output_lengths)
        ]
        tone_predictions = [
            tone_tokenizer.decode(ids[:length].cpu())
            for ids, length in zip(tone_ids, output_lengths)
        ]

        for row, jyutping_prediction, tone_prediction in zip(rows, jyutping_predictions, tone_predictions):
            tone_text = row.get(tone_column)
            jyutping_reference, tone_reference = annotation_to_legacy_texts(
                row[jyutping_column],
                tone_text=tone_text,
                annotation_type=annotation_type,
            )
            reference = decode_full_jyutping(jyutping_reference, tone_reference) or ""
            strict_prediction = decode_full_jyutping(jyutping_prediction, tone_prediction)
            tolerant_prediction, length_mismatch = combine_base_and_tones(
                jyutping_prediction, tone_prediction
            )
            predictions.append(
                Prediction(
                    reference=reference,
                    prediction=tolerant_prediction,
                    token_reference=jyutping_reference,
                    token_prediction=jyutping_prediction,
                    tone_reference=tone_reference,
                    tone_prediction=tone_prediction,
                    strict_prediction=strict_prediction,
                    length_mismatch=length_mismatch,
                )
            )

    return predictions


def predict_nucleus_tone(
    eval_dataset,
    model_id: str,
    base_model_id: str,
    vocab_path: str,
    jyutping_column: str,
    tone_column: str,
    annotation_type: str,
    batch_size: int,
) -> list[Prediction]:
    from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer

    from model import Wav2Vec2BertForJyutpingCTC

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    patch_added_tokens(tokenizer)

    feature_extractor_source = model_id if os.path.exists(model_id) else base_model_id
    try:
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(feature_extractor_source)
    except OSError:
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(base_model_id)
    processor = Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

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
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions = []
    for start in range(0, len(eval_dataset), batch_size):
        rows = eval_dataset.select(range(start, min(start + batch_size, len(eval_dataset))))
        decoded_audio = [read_audio(row["audio"]) for row in rows]
        audio_arrays = [audio_array for audio_array, _ in decoded_audio]
        sampling_rates = [sampling_rate for _, sampling_rate in decoded_audio]
        sampling_rate = sampling_rates[0]

        inputs = processor(audio_arrays, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.inference_mode():
            outputs = model(
                input_features=inputs.input_features.to(device),
                attention_mask=attention_mask,
                return_dict=True,
            )

        pred_ids = torch.argmax(outputs.logits, dim=-1)
        if attention_mask is not None:
            output_lengths = model._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)
        else:
            output_lengths = torch.full(
                (pred_ids.shape[0],),
                pred_ids.shape[1],
                device=pred_ids.device,
                dtype=torch.long,
            )
        token_predictions = [
            processor.decode(ids[:length].cpu())
            for ids, length in zip(pred_ids, output_lengths)
        ]

        for row, token_prediction in zip(rows, token_predictions):
            tone_text = row.get(tone_column)
            token_reference = annotation_to_nucleus_tone_text(
                row[jyutping_column],
                tone_text=tone_text,
                annotation_type=annotation_type,
            )
            predictions.append(
                Prediction(
                    reference=nucleus_tone_text_to_jyutping(token_reference),
                    prediction=nucleus_tone_text_to_jyutping(token_prediction),
                    token_reference=token_reference,
                    token_prediction=token_prediction,
                )
            )

    return predictions


def summarize(name: str, predictions: list[Prediction], examples: int) -> None:
    references = [prediction.reference for prediction in predictions]
    predicted = [prediction.prediction for prediction in predictions]
    counts = Counter()
    confusions = Counter()

    for prediction in predictions:
        sample_counts, sample_confusions = classify_ops(prediction.reference, prediction.prediction)
        counts.update(sample_counts)
        confusions.update(sample_confusions)

    total_ref_syllables = sum(len(reference.split()) for reference in references)
    total_edits = sum(value for key, value in counts.items() if key != "correct")
    substitutions = counts["tone_only"] + counts["base_only"] + counts["base_and_tone"]

    print()
    print(f"== {name} ==")
    print(f"samples: {len(predictions)}")
    print(f"syllables: {total_ref_syllables}")
    print(f"SER: {word_error_rate(references, predicted):.6f}")
    if any(prediction.strict_prediction is not None for prediction in predictions):
        strict_pairs = [
            (prediction.reference, prediction.strict_prediction)
            for prediction in predictions
            if prediction.strict_prediction is not None
        ]
        if strict_pairs:
            print(
                "strict SER on non-mismatched samples: "
                f"{word_error_rate([ref for ref, _ in strict_pairs], [pred for _, pred in strict_pairs]):.6f}"
            )
    mismatches = sum(prediction.length_mismatch for prediction in predictions)
    if mismatches:
        print(f"base/tone length mismatches: {mismatches}/{len(predictions)}")
    print(f"edits: {total_edits}")
    print(f"substitutions: {substitutions}")
    for key in ("tone_only", "base_only", "base_and_tone", "deletion", "insertion"):
        value = counts[key]
        percent = value / total_edits * 100 if total_edits else 0.0
        print(f"{key}: {value} ({percent:.1f}% of edits)")

    print("top confusions:")
    for (reference, prediction), count in confusions.most_common(10):
        print(f"  {reference} -> {prediction}: {count}")

    if examples:
        print("examples:")
        shown = 0
        for prediction in predictions:
            if prediction.reference == prediction.prediction:
                continue
            print(f"  ref:  {prediction.reference}")
            print(f"  pred: {prediction.prediction}")
            if prediction.tone_reference or prediction.tone_prediction:
                print(f"  base: {prediction.token_reference} -> {prediction.token_prediction}")
                print(f"  tone: {prediction.tone_reference} -> {prediction.tone_prediction}")
            else:
                print(f"  tok:  {prediction.token_reference} -> {prediction.token_prediction}")
            shown += 1
            if shown >= examples:
                break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published_model_id", default=DEFAULT_PUBLISHED_MODEL_ID)
    parser.add_argument("--checkpoint_model_id", default=DEFAULT_CHECKPOINT_MODEL_ID)
    parser.add_argument("--base_model_id", default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", choices=["test", "validation", "train"], default="test")
    parser.add_argument("--vocab_path", default="vocab_nucleus_tone.json")
    parser.add_argument("--jyutping_column", default="phone")
    parser.add_argument("--tone_column", default="tones")
    parser.add_argument("--annotation_type", choices=["auto", "inline", "separate"], default="inline")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()

    dataset_dict, eval_dataset = load_eval_dataset(args.dataset, args.split, args.max_samples, args.seed)
    print(f"dataset: {args.dataset}")
    print(
        "split sizes: "
        f"train={len(dataset_dict['train'])}, "
        f"validation={len(dataset_dict['validation'])}, "
        f"test={len(dataset_dict['test'])}"
    )
    print(f"evaluated split: {args.split} ({len(eval_dataset)} samples)")

    published = predict_published(
        eval_dataset,
        args.published_model_id,
        args.jyutping_column,
        args.tone_column,
        args.annotation_type,
        args.batch_size,
    )
    checkpoint = predict_nucleus_tone(
        eval_dataset,
        args.checkpoint_model_id,
        args.base_model_id,
        args.vocab_path,
        args.jyutping_column,
        args.tone_column,
        args.annotation_type,
        args.batch_size,
    )

    summarize(args.published_model_id, published, args.examples)
    summarize(args.checkpoint_model_id, checkpoint, args.examples)


if __name__ == "__main__":
    main()
