import sqlite3
import re
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import nltk

# Téléchargement stopwords si jamais ce n'est pas fait
nltk.download("stopwords")
nltk.download("punkt")
stop_words = set(stopwords.words("french"))

DB_PATH = "data/db/corpus.db"

def clean_text(text):
    """Nettoyer le texte pour tokenisation"""
    text = text.lower()
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s]", "", text)
    tokens = word_tokenize(text, language="french")
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

def main():
    # --- Charger les abstracts ---
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT abstract_clean FROM documents")
    abstracts = [row[0] for row in c.fetchall()]
    conn.close()

    # --- Prétraitement pour LDA ---
    tokenized_docs = [clean_text(doc) for doc in abstracts]

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
