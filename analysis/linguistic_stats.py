import sqlite3
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk import ngrams
import nltk

# Téléchargement stopwords si jamais ce n'est pas fait
nltk.download("stopwords")
stop_words = set(stopwords.words("french"))

DB_PATH = "data/db/corpus.db"

def clean_tokenize(text):
    """Nettoyer et tokeniser un texte"""
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return tokens

def main():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT abstract_clean FROM documents")
    abstracts = [row[0] for row in c.fetchall()]
    conn.close()

    total_docs = len(abstracts)
    all_tokens = []
    doc_lengths = []

    for text in abstracts:
        tokens = clean_tokenize(text)
        all_tokens.extend(tokens)
        doc_lengths.append(len(tokens))

    vocab = set(all_tokens)
    word_counts = Counter(all_tokens)
    bigram_counts = Counter(ngrams(all_tokens, 2))

    print(f"Nombre de documents : {total_docs}")
    print(f"Nombre total de mots : {sum(doc_lengths)}")
    print(f"Vocabulaire unique : {len(vocab)}")
    print("\n10 mots les plus fréquents :")
    for word, count in word_counts.most_common(10):
        print(f"{word}: {count}")
    print("\n10 bigrams les plus fréquents :")
    for bigram, count in bigram_counts.most_common(10):
        print(f"{' '.join(bigram)}: {count}")

if __name__ == "__main__":
    main()
