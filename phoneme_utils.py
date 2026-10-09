import re
from typing import List

# Cantonese (Jyutping) onsets — longest first to prevent prefix-shadowing
YUE_ONSETS = (
    "gw", "kw", "ng",
    "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "s", "t", "w", "z",
)
YUE_SYLLABIC_NASALS = frozenset({"m", "ng"})

# Mandarin (Pinyin) onsets — longest first (zh/ch/sh before z/c/s)
ZH_ONSETS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "r", "z", "c", "s", "y", "w",
)

TONED_YUE_RE = re.compile(r"^([a-z]+)([1-6])$")
TONED_ZH_RE = re.compile(r"^([a-z]+)([1-5])$")

_SKIP = frozenset({"<pad>", "<unk>", "<blank>", "[PAD]", "[UNK]"})

YUE_PREFIX = "yue_"
ZH_PREFIX = "zh_"


def split_jyutping_syllable(syllable: str) -> List[str]:
    """
    Split one toned Jyutping syllable into language-prefixed tokens.

    "gwong2" -> ["yue_gw", "yue_ong2"]
    "m4"     -> ["yue_m4"]
    "aa3"    -> ["yue_aa3"]
    """
    syllable = syllable.strip().lower()
    m = TONED_YUE_RE.match(syllable)
    if not m:
        return [f"{YUE_PREFIX}{syllable}"]
    base, tone = m.groups()
    if base in YUE_SYLLABIC_NASALS:
        return [f"{YUE_PREFIX}{base}{tone}"]
    for onset in YUE_ONSETS:
        if base.startswith(onset) and len(base) > len(onset):
            return [f"{YUE_PREFIX}{onset}", f"{YUE_PREFIX}{base[len(onset):]}{tone}"]
    return [f"{YUE_PREFIX}{base}{tone}"]


def split_pinyin_syllable(syllable: str) -> List[str]:
    """
    Split one toned Pinyin syllable into language-prefixed tokens.

    "zhong1" -> ["zh_zh", "zh_ong1"]
    "a1"     -> ["zh_a1"]
    "yi1"    -> ["zh_y", "zh_i1"]
    """
    syllable = syllable.strip().lower()
    m = TONED_ZH_RE.match(syllable)
    if not m:
        return [f"{ZH_PREFIX}{syllable}"]
    base, tone = m.groups()
    for onset in ZH_ONSETS:
        if base.startswith(onset) and len(base) > len(onset):
            return [f"{ZH_PREFIX}{onset}", f"{ZH_PREFIX}{base[len(onset):]}{tone}"]
    return [f"{ZH_PREFIX}{base}{tone}"]


def phonemes_to_tokens(phonemes: List[str], lang: str) -> List[str]:
    """
    Convert a raw phoneme list (from the dataset) into a unified token stream.

    - lang == "en":  ARPAbet tokens passed through as-is; "|" word boundaries
                     are already present in the phoneme list.
    - lang == "yue": each entry is a full Jyutping syllable; split into
                     onset + final+tone with yue_ prefix; "|" inserted between
                     syllables (one character = one syllable = one "word" unit).
    - lang == "zh":  same as yue but using Pinyin splitting and zh_ prefix.
    """
    tokens: List[str] = []

    if lang == "en":
        for ph in phonemes:
            if ph not in _SKIP:
                tokens.append(ph)
        return tokens

    for ph in phonemes:
        if ph in _SKIP:
            continue
        if ph in ("|", "."):
            # explicit boundary from dataset — absorb it, avoid double |
            if tokens and tokens[-1] != "|":
                tokens.append("|")
            continue
        # CJK syllable: insert | before it (unless at start or already after |)
        if tokens and tokens[-1] != "|":
            tokens.append("|")
        if lang == "yue":
            tokens.extend(split_jyutping_syllable(ph))
        elif lang == "zh":
            tokens.extend(split_pinyin_syllable(ph))
        else:
            tokens.append(ph)

    return tokens
