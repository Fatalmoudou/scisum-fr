import sqlite3
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models

DB_PATH = "data/db/corpus.db"
STOPWORDS_PATH = Path("analysis/stopwords_fr.txt")

_RE_TOKEN = re.compile(r"[a-zàâçéèêëîïôûùüÿñæœ]+", re.IGNORECASE)


def load_stopwords():
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def clean_text(text, stop_words):
    """Nettoyer le texte pour tokenisation."""
    tokens = [t.lower() for t in _RE_TOKEN.findall(text)]
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

def main():
    # --- Charger les abstracts ---
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT texte_nettoye FROM documents")
    abstracts = [row[0] for row in c.fetchall()]
    conn.close()

    # --- Prétraitement pour LDA ---
    stop_words = load_stopwords()
    tokenized_docs = [clean_text(doc, stop_words) for doc in abstracts]

    # --- Création dictionnaire & corpus pour LDA ---
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]

    # --- LDA ---
    num_topics = 5  # ajustable selon ton corpus
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    print("\n--- Thèmes extraits (LDA) ---")
    for idx, topic in lda_model.print_topics(num_words=5):
        print(f"Thème {idx+1}: {topic}")

    # --- TF-IDF vectorisation (optionnelle pour ML) ---
    docs_joined = [" ".join(doc) for doc in tokenized_docs]
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(docs_joined)
    print(f"\nTF-IDF shape: {tfidf_matrix.shape} (documents x termes)")

if __name__ == "__main__":
    main()
