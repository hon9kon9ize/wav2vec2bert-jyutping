"""
Quick smoke test for the multilingual phoneme checkpoint.
Loads the model from checkpoints_multilingual_phoneme_h100/checkpoint-4612
and runs inference on a short test audio or synthetic input.

Usage:
    python test_multilingual.py [--audio path/to/audio.wav]
"""
import argparse
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import numpy as np
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2BertProcessor,
    SeamlessM4TFeatureExtractor,
    AddedToken,
)
from model import Wav2Vec2BertForJyutpingCTC

DEFAULT_CHECKPOINT = "checkpoints_multilingual_phoneme_h100/checkpoint-4612"
BASE_MODEL = "facebook/w2v-bert-2.0"
SAMPLE_RATE = 16000


def patch_added_tokens(tokenizer) -> None:
    for key in tokenizer.get_vocab().keys():
        if key not in tokenizer.special_tokens_map.values():
            idx = tokenizer.get_vocab()[key]
            tokenizer._added_tokens_decoder[idx] = AddedToken(
                key, lstrip=False, rstrip=False
            )


def load_model(checkpoint: str):
    vocab_path = os.path.join(checkpoint, "vocab.json")

    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    patch_added_tokens(tokenizer)

    if os.path.exists(os.path.join(checkpoint, "preprocessor_config.json")):
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(checkpoint)
    else:
        feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(BASE_MODEL)

    processor = Wav2Vec2BertProcessor(
        feature_extractor=feature_extractor, tokenizer=tokenizer
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Wav2Vec2BertForJyutpingCTC.from_pretrained(
        checkpoint,
        ignore_mismatched_sizes=True,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
    )
    model.to(device).eval()
    print(f"Loaded model from {checkpoint}")
    print(f"  vocab size : {len(tokenizer)}")
    print(f"  device     : {device}")
    return model, processor, device


def run_inference(model, processor, device, audio: np.ndarray):
    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    input_features = inputs.input_features.to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        outputs = model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )

    pred_ids = torch.argmax(outputs.logits, dim=-1)
    if attention_mask is not None:
        lengths = model._get_feat_extract_output_lengths(
            attention_mask.sum(-1)
        ).to(torch.long)
    else:
        lengths = torch.full(
            (pred_ids.shape[0],), pred_ids.shape[1], dtype=torch.long
        )

    decoded = [
        processor.decode(ids[:l].cpu())
        for ids, l in zip(pred_ids, lengths)
    ]
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--audio", default=None, help="Path to a 16 kHz WAV/MP3 file")
    args = parser.parse_args()

    model, processor, device = load_model(args.checkpoint)

    if args.audio:
        import soundfile as sf
        audio, sr = sf.read(args.audio)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import resampy
            audio = resampy.resample(audio, sr, SAMPLE_RATE)
        audio = audio.astype(np.float32)
        print(f"\nAudio: {args.audio}  ({len(audio)/SAMPLE_RATE:.2f}s)")
    else:
        # Synthetic sine wave — just checks model loads and runs without error
        duration = 1.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(SAMPLE_RATE * duration))).astype(np.float32)
        print("\nNo audio provided — using synthetic 1s sine wave (output will be noise)")

    decoded = run_inference(model, processor, device, audio)
    print(f"\nRaw token output: {decoded[0]!r}")
    print("\nTest passed — model loaded and ran inference successfully.")


if __name__ == "__main__":
    main()
