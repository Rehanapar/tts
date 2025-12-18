

_pad = "_"
_space = " "

# Punctuation
_punctuation = list(".,!?;:—-…\"'")

# Digits
_digits = list("0123456789")

# Tamil Vowels (Uyir)
_uyir = list("அஆஇஈஉஊஎஏஐஒஓஔ")

# Tamil Consonants (Mei base)
_mei_base = list("கஙசஞடணதநபமயரலவழளறன")

# Tamil vowel signs (Uyrmai markers)
_vowel_signs = list("ாிீுூெேைொோௌ்")

# Generate Uyir-Mei combinations (correct way)
_uyir_mei = []
for cons in _mei_base:
    for sign in _vowel_signs:
        _uyir_mei.append(cons + sign)

# Special character
_special = ["ஃ"]

# FINAL SYMBOL SET
symbols = (
    [_pad, _space]
    + _punctuation
    + _digits
    + _uyir
    + _mei_base
    + _vowel_signs
    + _uyir_mei
    + _special
)

# Mapping
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Space ID
SPACE_ID = symbols.index(" ")

