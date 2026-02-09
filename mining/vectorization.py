from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def load_stopwords():
    path = Path("analysis/stopwords_fr.txt")
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_tfidf(corpus, min_df=2, max_df=0.9):
    stopwords = load_stopwords()

    vectorizer = TfidfVectorizer(
        stop_words=stopwords,
        min_df=min_df,
        max_df=max_df
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

# def build_count(corpus, min_df=2):
#     vectorizer = CountVectorizer(
#         stop_words="french",
#         min_df=min_df
#     )
#     X = vectorizer.fit_transform(corpus)
#     return X, vectorizer
