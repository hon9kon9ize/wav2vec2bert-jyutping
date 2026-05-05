import argparse
import json
from pathlib import Path

from dataset_utils import DEFAULT_DATASET, flatten_dataset, load_any_dataset
from jyutping import (
    ONSETS,
    SPECIAL_TOKENS,
    SYLLABIC_NASALS,
    TONES,
    annotation_to_nucleus_tone_text,
)


def default_tokens_from_base_vocab(vocab_path: str) -> set[str]:
    with open(vocab_path, encoding="utf-8") as f:
        base_vocab = json.load(f)

    tokens = set(ONSETS)
    for token in base_vocab:
        if token in SPECIAL_TOKENS or token in ONSETS:
            continue
        for tone in TONES:
            tokens.add(f"{token}{tone}")

    for nasal in SYLLABIC_NASALS:
        for tone in TONES:
            tokens.add(f"{nasal}{tone}")

    return tokens


def tokens_from_dataset(
    dataset_path: str,
    jyutping_column: str,
    tone_column: str,
    annotation_type: str,
) -> set[str]:
    ds = flatten_dataset(load_any_dataset(dataset_path))
    tokens = set()

    for row in ds:
        tone_text = row.get(tone_column)
        label_text = annotation_to_nucleus_tone_text(
            row[jyutping_column],
            tone_text=tone_text,
            annotation_type=annotation_type,
        )
        tokens.update(label_text.split())

    return tokens


def write_vocab(tokens: set[str], output_path: str) -> None:
    ordered_tokens = SPECIAL_TOKENS + sorted(tokens)
    vocab = {token: idx for idx, token in enumerate(ordered_tokens)}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--base_vocab", type=str, default="vocab.json")
    parser.add_argument("--output", type=str, default="vocab_nucleus_tone.json")
    parser.add_argument("--jyutping_column", type=str, default="phone")
    parser.add_argument("--tone_column", type=str, default="tones")
    parser.add_argument(
        "--annotation_type",
        choices=["auto", "inline", "separate"],
        default="inline",
    )
    args = parser.parse_args()

    if args.dataset:
        tokens = tokens_from_dataset(
            args.dataset,
            args.jyutping_column,
            args.tone_column,
            args.annotation_type,
        )
    else:
        tokens = default_tokens_from_base_vocab(args.base_vocab)

    write_vocab(tokens, args.output)
    print(f"Wrote {len(tokens) + len(SPECIAL_TOKENS)} tokens to {Path(args.output)}")


if __name__ == "__main__":
    main()
