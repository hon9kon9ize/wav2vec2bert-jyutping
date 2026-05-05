import re
from typing import Iterable, List


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "|"]
TONES = tuple("123456")
ONSETS = (
    "gw",
    "kw",
    "ng",
    "b",
    "c",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "s",
    "t",
    "w",
    "z",
)
SYLLABIC_NASALS = {"m", "ng"}
JYUTPING_RE = re.compile(r"^([a-z]+)([1-6])$")
PUNCTUATION_RE = re.compile(r"^[^\w]+$")


def normalize_jyutping_text(text: str) -> str:
    tokens = []
    for token in text.strip().lower().split():
        token = token.strip()
        if not token or PUNCTUATION_RE.match(token):
            continue
        token = token.strip(".,!?;:\"'()[]{}<>，。！？；：「」『』、")
        if token:
            tokens.append(token)

    return " ".join(tokens)


def split_nucleus_tone_syllable(syllable: str) -> List[str]:
    """Split one toned Jyutping syllable into onset and rime+tone tokens."""
    syllable = syllable.strip().lower()
    match = JYUTPING_RE.match(syllable)
    if not match:
        raise ValueError(f"Invalid toned Jyutping syllable: {syllable!r}")

    base, tone = match.groups()
    if base in SYLLABIC_NASALS:
        return [f"{base}{tone}"]

    for onset in ONSETS:
        if base.startswith(onset) and len(base) > len(onset):
            return [onset, f"{base[len(onset):]}{tone}"]

    return [f"{base}{tone}"]


def jyutping_to_nucleus_tone_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for syllable in normalize_jyutping_text(text).split():
        tokens.extend(split_nucleus_tone_syllable(syllable))
    return tokens


def jyutping_to_nucleus_tone_text(text: str) -> str:
    return " ".join(jyutping_to_nucleus_tone_tokens(text))


def split_toned_jyutping_to_legacy_tokens(text: str) -> tuple[str, str]:
    jyutping_tokens: List[str] = []
    tone_tokens: List[str] = []

    for syllable in normalize_jyutping_text(text).split():
        syllable = syllable.strip().lower()
        match = JYUTPING_RE.match(syllable)
        if not match:
            raise ValueError(f"Invalid toned Jyutping syllable: {syllable!r}")

        base, tone = match.groups()
        tone_tokens.append(tone)

        if base in SYLLABIC_NASALS:
            jyutping_tokens.append(base)
            continue

        matched_onset = ""
        for onset in ONSETS:
            if base.startswith(onset) and len(base) > len(onset):
                matched_onset = onset
                break

        if matched_onset:
            jyutping_tokens.extend([matched_onset, base[len(matched_onset) :]])
        else:
            jyutping_tokens.append(base)

    return " ".join(jyutping_tokens), " ".join(tone_tokens)


def tone_less_tokens_to_jyutping(text: str) -> str:
    pieces = []
    for token in normalize_jyutping_text(text).split():
        if token in ONSETS:
            pieces.append(f"_{token}")
        else:
            pieces.append(f"{token}_")

    return re.sub(r"\s+", " ", "".join(pieces).replace("_", " ").strip())


def add_tones_to_jyutping(jyutping_text: str, tone_text: str) -> str:
    jyutping_syllables = jyutping_text.split()
    tones = tone_text.split()

    if len(tones) == 1 and len(jyutping_syllables) > 1 and len(tones[0]) > 1:
        tones = list(tones[0])

    if len(jyutping_syllables) != len(tones):
        raise ValueError(
            "Jyutping and tone annotation lengths differ: "
            f"{len(jyutping_syllables)} syllables vs {len(tones)} tones"
        )

    return " ".join(
        f"{syllable}{tone}" for syllable, tone in zip(jyutping_syllables, tones)
    )


def separate_tone_annotation_to_jyutping(jyutping_text: str, tone_text: str) -> str:
    if re.search(r"[1-6]", jyutping_text):
        return jyutping_text

    tokenized = tone_less_tokens_to_jyutping(jyutping_text)
    return add_tones_to_jyutping(tokenized, tone_text)


def annotation_to_legacy_texts(
    jyutping_text: str, tone_text: str = None, annotation_type: str = "auto"
) -> tuple[str, str]:
    if annotation_type not in {"auto", "inline", "separate"}:
        raise ValueError(f"Unsupported annotation_type: {annotation_type!r}")

    if annotation_type == "inline" or (
        annotation_type == "auto" and re.search(r"[1-6]", jyutping_text)
    ):
        return split_toned_jyutping_to_legacy_tokens(jyutping_text)

    if tone_text is None:
        raise ValueError("tone_text is required for separate tone annotations")

    return jyutping_text, tone_text


def annotation_to_nucleus_tone_text(
    jyutping_text: str, tone_text: str = None, annotation_type: str = "auto"
) -> str:
    if annotation_type not in {"auto", "inline", "separate"}:
        raise ValueError(f"Unsupported annotation_type: {annotation_type!r}")

    if annotation_type == "inline" or (
        annotation_type == "auto" and re.search(r"[1-6]", jyutping_text)
    ):
        toned_jyutping = jyutping_text
    else:
        if tone_text is None:
            raise ValueError("tone_text is required for separate tone annotations")
        toned_jyutping = separate_tone_annotation_to_jyutping(jyutping_text, tone_text)

    return jyutping_to_nucleus_tone_text(toned_jyutping)


def nucleus_tone_tokens_to_jyutping(tokens: Iterable[str]) -> str:
    syllables: List[str] = []
    onset = ""

    for token in tokens:
        token = token.strip()
        if not token or token in SPECIAL_TOKENS:
            continue

        if token in ONSETS:
            if onset:
                syllables.append(onset)
            onset = token
            continue

        if JYUTPING_RE.match(token):
            syllables.append(f"{onset}{token}")
            onset = ""
            continue

        if onset:
            syllables.append(onset)
            onset = ""
        syllables.append(token)

    if onset:
        syllables.append(onset)

    return " ".join(syllables)


def nucleus_tone_text_to_jyutping(text: str) -> str:
    return nucleus_tone_tokens_to_jyutping(text.split())
