import sqlite3
import re
from collections import Counter
from pathlib import Path

DB_PATH = "data/db/corpus.db"
STOPWORDS_PATH = Path("analysis/stopwords_fr.txt")

_RE_TOKEN = re.compile(r"[a-zàâçéèêëîïôûùüÿñæœ]+", re.IGNORECASE)


def load_stopwords():
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def clean_tokenize(text, stop_words):
    """Nettoyer et tokeniser un texte."""
    tokens = [t.lower() for t in _RE_TOKEN.findall(text)]
    return [t for t in tokens if t not in stop_words]

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT texte_nettoye FROM documents")
    abstracts = [row[0] for row in c.fetchall()]
    conn.close()

    total_docs = len(abstracts)
    all_tokens = []
    doc_lengths = []
    stop_words = load_stopwords()

    for text in abstracts:
        tokens = clean_tokenize(text, stop_words)
        all_tokens.extend(tokens)
        doc_lengths.append(len(tokens))

    vocab = set(all_tokens)
    word_counts = Counter(all_tokens)
    bigram_counts = Counter(zip(all_tokens, all_tokens[1:]))
    trigram_counts = Counter(zip(all_tokens, all_tokens[1:], all_tokens[2:]))
    hapax = sum(1 for w, c in word_counts.items() if c == 1)

    print(f"Nombre de documents : {total_docs}")
    print(f"Nombre total de mots : {sum(doc_lengths)}")
    print(f"Vocabulaire unique : {len(vocab)}")
    if total_docs:
        print(f"Longueur moyenne des documents : {sum(doc_lengths)/total_docs:.1f} mots")
    if vocab:
        print(f"Ratio hapax/vocabulaire : {hapax/len(vocab):.3f}")

    print("\n50 mots les plus fréquents :")
    for word, count in word_counts.most_common(50):
        print(f"{word}: {count}")
    print("\n10 bigrams les plus fréquents :")
    for bigram, count in bigram_counts.most_common(10):
        print(f"{' '.join(bigram)}: {count}")
    print("\n10 trigrams les plus fréquents :")
    for trigram, count in trigram_counts.most_common(10):
        print(f"{' '.join(trigram)}: {count}")

if __name__ == "__main__":
    main()
