"""text/__init__.py — Tamil VITS adapted for 247-letter alphabet"""

from text import cleaners
from text.symbols import symbols

# ---------------------- Symbol Mappings ----------------------
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# ---------------------- Sequence Conversion ----------------------
def text_to_sequence(text, cleaner_names):
    """Converts a string of text to a sequence of IDs corresponding to the symbols."""
    sequence = []

    clean_text = _clean_text(text, cleaner_names)
    for symbol in clean_text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
        else:
            # Map unknown symbols to _pad (or skip)
            print(f"Warning: unknown symbol '{symbol}' mapped to _pad")
            sequence.append(_symbol_to_id["_"])
    return sequence


def cleaned_text_to_sequence(cleaned_text):
    """Converts pre-cleaned text to sequence of symbol IDs."""
    sequence = []
    for symbol in cleaned_text:
        if symbol in _symbol_to_id:
            sequence.append(_symbol_to_id[symbol])
        else:
            sequence.append(_symbol_to_id["_"])
    return sequence


def sequence_to_text(sequence):
    """Converts a sequence of IDs back to a string."""
    result = ""
    for symbol_id in sequence:
        if symbol_id in _id_to_symbol:
            result += _id_to_symbol[symbol_id]
        else:
            result += "_"  # fallback
    return result


# ---------------------- Text Cleaning ----------------------
def _clean_text(text, cleaner_names):
    """Run text through the sequence of cleaner functions."""
    for name in cleaner_names:
        if not hasattr(cleaners, name):
            raise Exception(f"Unknown cleaner: {name}")
        cleaner = getattr(cleaners, name)
        text = cleaner(text)
    return text
