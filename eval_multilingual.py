"""
Evaluation script for the multilingual phoneme CTC model.
Reports PER (token-level) and SER (syllable-level, decoded to full jyutping)
on the same 500-sample test split used by eval_w2v2bert_nucleus_tone.py.

Usage:
    python eval_multilingual.py \
        --model_id checkpoints_multilingual_phoneme_h100/checkpoint-4612 \
        --vocab_path vocab_multilingual_phoneme.json
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import torch
from dataset_utils import DEFAULT_DATASET, load_any_dataset, make_train_validation_test_split
from phoneme_utils import phonemes_to_tokens

DEFAULT_MODEL_ID = "checkpoints_multilingual_phoneme_h100/checkpoint-4612"
DEFAULT_BASE_MODEL_ID = "facebook/w2v-bert-2.0"


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

    for key in tokenizer.get_vocab().keys():
        if key not in tokenizer.special_tokens_map.values():
            idx = tokenizer.get_vocab()[key]
            tokenizer._added_tokens_decoder[idx] = AddedToken(
                key, lstrip=False, rstrip=False
            )


def read_audio(audio) -> tuple:
    if audio.get("array") is not None:
        return np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"]
    if audio.get("bytes") is not None:
        import io
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(audio["bytes"]))
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    raise ValueError("Cannot read audio from row")


def jyutping_text_to_multilingual_tokens(jyutping_text: str) -> str:
    """
    Convert inline toned jyutping ('maa4 maa1 go3') to the multilingual
    token string ('yue_m yue_aa4 | yue_m yue_aa1 | yue_g yue_o3').
    """
    syllables = jyutping_text.strip().split()
    tokens = phonemes_to_tokens(syllables, "yue")
    return " ".join(tokens)


def multilingual_tokens_to_jyutping(text: str) -> str:
    """
    Reverse: 'yue_m yue_aa4 | yue_m yue_aa1 | yue_g yue_o3' -> 'maa4 maa1 go3'
    Groups consecutive yue_* tokens between '|' separators into jyutping syllables.
    Non-yue tokens (zh_*, ARPAbet) are ignored for Cantonese-only evaluation.
    """
    syllables = []
    current = []
    for token in text.split():
        if token == "|":
            if current:
                syllables.append("".join(current))
                current = []
        elif token.startswith("yue_"):
            current.append(token[4:])  # strip 'yue_' prefix
    if current:
        syllables.append("".join(current))
    return " ".join(syllables)


def evaluate(
    model_id: str,
    base_model_id: str,
    dataset: str,
    split: str,
    vocab_path: str,
    jyutping_column: str,
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
    from model import Wav2Vec2BertForJyutpingCTC

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
        vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    patch_added_tokens(tokenizer)

    feature_extractor_source = model_id if os.path.exists(model_id) else base_model_id
    try:
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(
            feature_extractor_source
        )
    except OSError:
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(base_model_id)

    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    model = Wav2Vec2BertForJyutpingCTC.from_pretrained(
        model_id,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
        add_adapter=False,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"device: {device}", flush=True)

    token_predictions = []
    token_references = []
    jyutping_predictions = []
    jyutping_references = []

    for start in range(0, len(eval_dataset), batch_size):
        rows = eval_dataset.select(
            range(start, min(start + batch_size, len(eval_dataset)))
        )
        decoded_audio = [read_audio(row["audio"]) for row in rows]
        audio_arrays = [a for a, _ in decoded_audio]
        sampling_rates = [sr for _, sr in decoded_audio]
        sampling_rate = sampling_rates[0]

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

        pred_ids = torch.argmax(outputs.logits, dim=-1)
        if attention_mask is not None:
            output_lengths = model._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)
        else:
            output_lengths = torch.full(
                (pred_ids.shape[0],), pred_ids.shape[1], dtype=torch.long
            )
        batch_token_predictions = [
            processor.decode(ids[:length].cpu())
            for ids, length in zip(pred_ids, output_lengths)
        ]

        for row, token_prediction in zip(rows, batch_token_predictions):
            token_reference = jyutping_text_to_multilingual_tokens(row[jyutping_column])
            token_predictions.append(token_prediction)
            token_references.append(token_reference)
            jyutping_predictions.append(multilingual_tokens_to_jyutping(token_prediction))
            jyutping_references.append(multilingual_tokens_to_jyutping(token_reference))

        done = min(start + batch_size, len(eval_dataset))
        print(f"  {done}/{len(eval_dataset)}", flush=True)

    per = word_error_rate(token_references, token_predictions)
    ser = word_error_rate(jyutping_references, jyutping_predictions)

    print(f"\nmodel: {model_id}")
    print(f"dataset: {dataset}")
    print(
        "split sizes: "
        f"train={len(dataset_dict['train'])}, "
        f"validation={len(dataset_dict['validation'])}, "
        f"test={len(dataset_dict['test'])}"
    )
    print(f"evaluated split: {split} ({len(eval_dataset)} samples)")
    print(f"PER (token-level): {per:.6f}")
    print(f"SER (syllable-level): {ser:.6f}")

    # Print a few examples
    print("\n--- Sample predictions ---")
    for ref_j, pred_j, ref_t, pred_t in list(zip(
        jyutping_references, jyutping_predictions, token_references, token_predictions
    ))[:5]:
        print(f"  ref  jyutping : {ref_j}")
        print(f"  pred jyutping : {pred_j}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split", choices=["test", "validation", "train"], default="test"
    )
    parser.add_argument("--vocab_path", type=str, default="vocab_multilingual_phoneme.json")
    parser.add_argument("--jyutping_column", type=str, default="phone")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        dataset=args.dataset,
        split=args.split,
        vocab_path=args.vocab_path,
        jyutping_column=args.jyutping_column,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
    )
