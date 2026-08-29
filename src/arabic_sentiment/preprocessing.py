import re


_ARABIC_DIACRITICS = re.compile(
    r"""
    ّ    | # Shadda
    َ    | # Fatha
    ً    | # Tanwin Fath
    ُ    | # Damma
    ٌ    | # Tanwin Damm
    ِ    | # Kasra
    ٍ    | # Tanwin Kasr
    ْ    | # Sukun
    ـ      # Tatweel
    """,
    re.VERBOSE,
)


def normalize_arabic_text(text: str) -> str:
    """Apply lightweight normalization to Arabic text."""

    text = text.strip()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove Arabic diacritics and tatweel
    text = re.sub(_ARABIC_DIACRITICS, "", text)

    # Normalize common Arabic letter variants
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ى", "ي", text)

    # Normalize repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def preprocess_text(text: str) -> str:
    """Validate and preprocess a single Arabic review."""

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if not text.strip():
        raise ValueError("Text cannot be empty.")

    return normalize_arabic_text(text)