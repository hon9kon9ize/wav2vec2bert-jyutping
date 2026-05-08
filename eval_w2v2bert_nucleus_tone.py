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
from jyutping import annotation_to_nucleus_tone_text, nucleus_tone_text_to_jyutping


DEFAULT_MODEL_ID = "checkpoints_nucleus_tone"
DEFAULT_BASE_MODEL_ID = "facebook/w2v-bert-2.0"


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


def pad_f0_features(features: list[list[float]], device: torch.device) -> torch.Tensor:
    tensors = [torch.tensor(feature, dtype=torch.float32) for feature in features]
    max_length = max(tensor.shape[0] for tensor in tensors)
    padded = [
        torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[0]))
        for tensor in tensors
    ]
    return torch.stack(padded).to(device)


def evaluate(
    model_id: str,
    base_model_id: str,
    dataset: str,
    split: str,
    vocab_path: str,
    jyutping_column: str,
    tone_column: str,
    annotation_type: str,
    batch_size: int,
    max_samples: int,
    seed: int,
    use_f0: bool,
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    token_predictions = []
    token_references = []
    jyutping_predictions = []
    jyutping_references = []

    for start in range(0, len(eval_dataset), batch_size):
        rows = eval_dataset.select(
            range(start, min(start + batch_size, len(eval_dataset)))
        )
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

        f0_features = None
        if use_f0:
            f0_features = pad_f0_features(
                [
                    extract_f0_features(audio_array, sampling_rate)
                    for audio_array in audio_arrays
                ],
                device,
            )

        with torch.inference_mode():
            outputs = model(
                input_features=input_features,
                attention_mask=attention_mask,
                f0_features=f0_features,
                return_dict=True,
            )

        pred_ids = torch.argmax(outputs.logits, dim=-1).cpu()
        batch_token_predictions = processor.batch_decode(pred_ids)

        for row, token_prediction in zip(rows, batch_token_predictions):
            tone_text = row.get(tone_column)
            token_reference = annotation_to_nucleus_tone_text(
                row[jyutping_column],
                tone_text=tone_text,
                annotation_type=annotation_type,
            )

            token_predictions.append(token_prediction)
            token_references.append(token_reference)
            jyutping_predictions.append(nucleus_tone_text_to_jyutping(token_prediction))
            jyutping_references.append(nucleus_tone_text_to_jyutping(token_reference))

    per = word_error_rate(token_references, token_predictions)
    ser = word_error_rate(jyutping_references, jyutping_predictions)

    print(f"model: {model_id}")
    print(f"dataset: {dataset}")
    print(
        "split sizes: "
        f"train={len(dataset_dict['train'])}, "
        f"validation={len(dataset_dict['validation'])}, "
        f"test={len(dataset_dict['test'])}"
    )
    print(f"evaluated split: {split} ({len(eval_dataset)} samples)")
    print(f"PER: {per:.6f}")
    print(f"SER: {ser:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--base_model_id", type=str, default=DEFAULT_BASE_MODEL_ID)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument(
        "--split", choices=["test", "validation", "train"], default="test"
    )
    parser.add_argument("--vocab_path", type=str, default="vocab_nucleus_tone.json")
    parser.add_argument("--jyutping_column", type=str, default="phone")
    parser.add_argument("--tone_column", type=str, default="tones")
    parser.add_argument(
        "--annotation_type",
        choices=["auto", "inline", "separate"],
        default="inline",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use_f0",
        action="store_true",
        help="Use this for checkpoints trained with --use_f0.",
    )
    args = parser.parse_args()

    evaluate(
        args.model_id,
        args.base_model_id,
        args.dataset,
        args.split,
        args.vocab_path,
        args.jyutping_column,
        args.tone_column,
        args.annotation_type,
        args.batch_size,
        args.max_samples,
        args.seed,
        args.use_f0,
    )
