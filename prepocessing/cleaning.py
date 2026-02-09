import re

_RE_ABSTRACT = re.compile(r"\bABSTRACT\b|\bAbstract\b")
_RE_KEYWORDS = re.compile(r"\bMOTS[- ]CL[ÉE]S\b|\bKeywords\b", re.IGNORECASE)
_RE_MATH = re.compile(r"\$[^$]+\$|\\\([^)]*\\\)|\\\[[^\]]*\\\]")
_RE_REFS = re.compile(r"\b(R[ée]f[ée]rences|Bibliographie)\b", re.IGNORECASE)


def clean_abstract(text):
    """
    Nettoie un résumé scientifique.
    - supprime les parties en anglais
    - supprime les sections MOTS-CLÉS / ABSTRACT
    - supprime les formules mathématiques simples
    - normalise les espaces
    """
    if not text:
        return ""

    text = _RE_ABSTRACT.split(text)[0]
    text = _RE_KEYWORDS.split(text)[0]
    text = _RE_REFS.split(text)[0]
    text = _RE_MATH.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
