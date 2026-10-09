#!/usr/bin/env python3
"""
Generate vocab_multilingual_phoneme.json from per-language vocab files.

Token design
------------
[PAD], [UNK], |  — shared CTC/delimiter specials
yue_<token>      — Cantonese onsets and finals+tone  (e.g. yue_b, yue_aa1, yue_m4)
zh_<token>       — Mandarin onsets and finals+tone   (e.g. zh_zh, zh_ong1, zh_a1)
<ARPABET>        — English phonemes, uppercase        (e.g. AA, CH, ER)

Cantonese and Mandarin tokens are always prefixed even when their romanisation
overlaps (e.g. yue_ang1 ≠ zh_ang1 — different vowel quality and tone contour).
"""

import json
import re
from pathlib import Path

from phoneme_utils import YUE_ONSETS, YUE_SYLLABIC_NASALS, TONED_ZH_RE, ZH_ONSETS

SCRIPT_DIR = Path(__file__).parent
PHONEME_DATASET_DIR = Path(
    "/home/pj24001684/ku40000295/jc/projects/phoneme_asr_dataset/output/hf"
)

_ZH_SKIP = frozenset({"<pad>", "<unk>", "<blank>", "."})
_EN_SKIP = frozenset({"<pad>", "<unk>", "<blank>", "|"})
_YUE_SKIP = frozenset({"[PAD]", "[UNK]", "|"})


def _derive_zh_tokens(zh_vocab: dict) -> "list[str]":
    """Return sorted list of unique zh_-prefixed onset and final+tone tokens."""
    tokens: set[str] = set()
    for syllable in zh_vocab:
        if syllable in _ZH_SKIP:
            continue
        m = TONED_ZH_RE.match(syllable.lower())
        if not m:
            continue
        base, tone = m.groups()
        matched = False
        for onset in ZH_ONSETS:
            if base.startswith(onset) and len(base) > len(onset):
                tokens.add(f"zh_{onset}")
                tokens.add(f"zh_{base[len(onset):]}{tone}")
                matched = True
                break
        if not matched:
            tokens.add(f"zh_{base}{tone}")
    return sorted(tokens)


def build_vocab() -> dict:
    vocab = {"[PAD]": 0, "[UNK]": 1, "|": 2}  # type: dict
    idx = 3

    # ── Cantonese: prefix every non-special token from vocab_nucleus_tone.json ──
    with open(SCRIPT_DIR / "vocab_nucleus_tone.json") as f:
        yue_vocab = json.load(f)

    for token in yue_vocab:
        if token in _YUE_SKIP:
            continue
        prefixed = f"yue_{token}"
        if prefixed not in vocab:
            vocab[prefixed] = idx
            idx += 1

    # ── Mandarin: derive onset + final+tone tokens from vocab_zh.json ──────────
    with open(PHONEME_DATASET_DIR / "vocab_zh.json") as f:
        zh_vocab = json.load(f)

    for token in _derive_zh_tokens(zh_vocab):
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    # ── English: uppercase ARPAbet from vocab_en.json ───────────────────────────
    with open(PHONEME_DATASET_DIR / "vocab_en.json") as f:
        en_vocab = json.load(f)

    for token in en_vocab:
        if token in _EN_SKIP:
            continue
        if token not in vocab:
            vocab[token] = idx
            idx += 1

    return vocab


if __name__ == "__main__":
    vocab = build_vocab()
    out = SCRIPT_DIR / "vocab_multilingual_phoneme.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(vocab)} tokens → {out}")

    # quick sanity check
    assert vocab["[PAD]"] == 0
    assert vocab["[UNK]"] == 1
    assert vocab["|"] == 2
    assert "yue_gw" in vocab
    assert "yue_aa1" in vocab
    assert "yue_m4" in vocab
    assert "zh_zh" in vocab
    assert "zh_ong1" in vocab
    assert "AA" in vocab
    assert "CH" in vocab
    print("Sanity checks passed.")
