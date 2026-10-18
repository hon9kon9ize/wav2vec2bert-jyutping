"""Toned Jyutping -> IPA (broad, phonemic; Hong Kong Cantonese).

The mapping is a lookup over the 19 onsets and 55 rimes of `vocab_nucleus_tone.json`,
following the standard Jyutping/IPA correspondence (Wikipedia "Jyutping", Hong Kong
inventory). Stop codas are written unreleased (p̚ t̚ k̚). Tones are Chao letters; the
entering tones (stop-final 1/3/6) have the same pitch as the level tones 1/3/6.

Not modelled: the glottal stop before onset-less syllables (`aa3` -> aː˧, not ʔaː˧),
the free variation of T1 (55 ~ 53) and T4 (21 ~ 11), and the n/l, ng/-, gw/g tone-merger
variants (the model's output is converted as given).
"""

import re
import sys
from typing import Iterable

from jyutping import JYUTPING_RE, ONSETS, SPECIAL_TOKENS, nucleus_tone_text_to_jyutping

ONSET_IPA = {
    "b": "p", "p": "pʰ", "d": "t", "t": "tʰ", "g": "k", "k": "kʰ",
    "gw": "kʷ", "kw": "kʷʰ", "z": "ts", "c": "tsʰ", "s": "s", "f": "f",
    "h": "h", "m": "m", "n": "n", "ng": "ŋ", "l": "l", "j": "j", "w": "w",
}  # fmt: skip

RIME_IPA = {
    "a": "ɐ",  # not a standard final; only `a6` (filler 誒) and `za6` occur, 7 of 2.46M labels
    "aa": "aː", "aai": "aːi", "aau": "aːu", "aam": "aːm", "aan": "aːn", "aang": "aːŋ",
    "aap": "aːp̚", "aat": "aːt̚", "aak": "aːk̚",
    "ai": "ɐi", "au": "ɐu", "am": "ɐm", "an": "ɐn", "ang": "ɐŋ",
    "ap": "ɐp̚", "at": "ɐt̚", "ak": "ɐk̚",
    "e": "ɛː", "ei": "ei", "eu": "ɛːu", "eng": "ɛːŋ", "ek": "ɛːk̚",
    "i": "iː", "iu": "iːu", "im": "iːm", "in": "iːn", "ing": "ɪŋ",
    "ip": "iːp̚", "it": "iːt̚", "ik": "ɪk̚",
    "o": "ɔː", "oi": "ɔːi", "ou": "ou", "on": "ɔːn", "ong": "ɔːŋ", "ot": "ɔːt̚", "ok": "ɔːk̚",
    "u": "uː", "ui": "uːi", "un": "uːn", "ung": "ʊŋ", "ut": "uːt̚", "uk": "ʊk̚",
    "oe": "œː", "oeng": "œːŋ", "oek": "œːk̚", "eoi": "ɵy", "eon": "ɵn", "eot": "ɵt̚",
    "yu": "yː", "yun": "yːn", "yut": "yːt̚",
    "m": "m̩", "ng": "ŋ̩",
}  # fmt: skip

TONE_IPA = {"1": "˥", "2": "˧˥", "3": "˧", "4": "˨˩", "5": "˩˧", "6": "˨"}


def syllable_to_ipa(syllable: str, tones: bool = True) -> str:
    """One toned Jyutping syllable (`zung3`) -> IPA (`tsʊŋ˧`)."""
    match = JYUTPING_RE.match(syllable.strip().lower())
    if not match:
        raise ValueError(f"Invalid toned Jyutping syllable: {syllable!r}")
    base, tone = match.groups()

    onset = ""
    rime = base
    if base not in RIME_IPA:  # `m`/`ng` alone are syllabic nasals, not onsets
        for candidate in ONSETS:
            if base.startswith(candidate) and base[len(candidate) :] in RIME_IPA:
                onset, rime = candidate, base[len(candidate) :]
                break
    if rime not in RIME_IPA:
        raise ValueError(f"Unknown Jyutping rime in {syllable!r}")

    return ONSET_IPA.get(onset, "") + RIME_IPA[rime] + (TONE_IPA[tone] if tones else "")


def jyutping_to_ipa(text: str, tones: bool = True) -> str:
    """Space-separated toned Jyutping -> space-separated IPA syllables.

    A piece that is a bare onset (a dangling onset in raw model output) becomes its IPA
    onset; any other unparseable piece is kept verbatim, so garbage output never raises.
    """
    out = []
    for piece in text.split():
        if piece in SPECIAL_TOKENS:
            continue
        try:
            out.append(syllable_to_ipa(piece, tones))
        except ValueError:
            out.append(ONSET_IPA.get(piece, piece))
    return " ".join(out)


def token_text_to_ipa(token_text: str, tones: bool = True) -> str:
    """Decoded model output (`z ung3 s o2`, from `ctc_ids_to_token_text`) -> IPA."""
    return jyutping_to_ipa(nucleus_tone_text_to_jyutping(token_text), tones)


def tokens_to_ipa(tokens: Iterable[str], tones: bool = True) -> str:
    return token_text_to_ipa(" ".join(tokens), tones)


if __name__ == "__main__":
    # python jyutping_ipa.py "zung3 so2 zau1 zi1"   (toned Jyutping)
    # python jyutping_ipa.py --tokens "z ung3 s o2"  (model token text)
    args = sys.argv[1:]
    convert = jyutping_to_ipa
    if args and args[0] == "--tokens":
        convert, args = token_text_to_ipa, args[1:]
    print(convert(" ".join(args) if args else sys.stdin.read()))
