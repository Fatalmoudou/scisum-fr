import re

def clean_abstract(text):
    """
    Nettoie un résumé scientifique :
    - supprime les parties en anglais
    - supprime les sections MOTS-CLÉS / ABSTRACT
    - normalise les espaces
    """
    if not text:
        return ""

    # Supprimer tout ce qui suit ABSTRACT ou ABSTRACT.
    text = re.split(r"\bABSTRACT\b|\bAbstract\b", text)[0]

    # Supprimer les sections mots-clés
    text = re.split(r"\bMOTS[- ]CLÉS\b|\bKeywords\b", text)[0]

    # Suppression des retours multiples
    text = re.sub(r"\s+", " ", text)

    return text.strip()
