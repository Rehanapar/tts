import re
import unicodedata
from .symbols import symbols

# Fast lookup
_valid_chars = set(symbols)

# Base consonants (MUST match mei_base)
_base_consonants = [
    "க","ங","ச","ஞ","ட","ண",
    "த","ந","ப","ம","ய","ர",
    "ல","வ","ழ","ள","ற","ன"
]

# Map vowel diacritics → independent uyir vowel
KOMBU_TO_UYIR = {
    "ா": "ஆ",
    "ி": "இ",
    "ீ": "ஈ",
    "ு": "உ",
    "ூ": "ஊ",
    "ெ": "எ",
    "ே": "ஏ",
    "ை": "ஐ",
    "ொ": "ஒ",
    "ோ": "ஓ",
    "ௌ": "ஔ",
}

_whitespace_re = re.compile(r"\s+")


def tamil_cleaner(text):
    """Normalize Tamil and convert uyirmei → consonant + uyir vowel."""
    text = unicodedata.normalize("NFC", text)

    result = ""
    i = 0

    while i < len(text):
        ch = text[i]

        # consonant + diacritic → mei + uyir
        if ch in _base_consonants and i + 1 < len(text):
            next_ch = text[i+1]
            if next_ch in KOMBU_TO_UYIR:
                result += ch + KOMBU_TO_UYIR[next_ch]
                i += 2
                continue

        result += ch
        i += 1

    # Remove characters that are NOT in symbol list
    cleaned = "".join(c if c in _valid_chars else " " for c in result)

    # Collapse multiple spaces
    cleaned = _whitespace_re.sub(" ", cleaned).strip()

    return cleaned

