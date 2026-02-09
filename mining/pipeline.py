import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from gensim import corpora, models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from wordcloud import WordCloud

_RE_TOKEN = re.compile(r"[a-zàâçéèêëîïôûùüÿñæœ]+", re.IGNORECASE)
STOPWORDS_PATH = Path("analysis/stopwords_fr.txt")


def load_stopwords():
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_corpus(path="data/cleaned/hal_dumas_metadata_cleaned.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = [
        doc.get("texte_nettoye") or doc.get("abstract_clean") or ""
        for doc in data
    ]
    texts = [t for t in texts if t]
    tokens = [_RE_TOKEN.findall(text.lower()) for text in texts]
    return texts, tokens


def build_tfidf(corpus, stop_words, min_df=2, max_df=0.9):
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        min_df=min_df,
        max_df=max_df
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def plot_elbow(k_values, inertias, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Inertie")
    plt.title("Méthode du coude")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_silhouette(k_values, scores, out_path):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, scores, marker="o")
    plt.xlabel("Nombre de clusters (k)")
    plt.ylabel("Score silhouette")
    plt.title("Score silhouette par k")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_tsne(tsne_coords, labels, out_path):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_coords[:, 0], tsne_coords[:, 1], c=labels, cmap="tab10", s=18)
    plt.legend(*scatter.legend_elements(), title="Clusters", loc="best")
    plt.title("t-SNE des documents")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_lda_topics(lda_model, out_path, num_words=8):
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    n_topics = len(topics)
    cols = 2
    rows = (n_topics + 1) // cols
    plt.figure(figsize=(10, 4 * rows))
    for i, (topic_id, words) in enumerate(topics, start=1):
        plt.subplot(rows, cols, i)
        labels = [w for w, _ in words]
        weights = [wgt for _, wgt in words]
        plt.barh(labels[::-1], weights[::-1])
        plt.title(f"Thème {topic_id + 1}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run_pipeline(
    input_path="data/cleaned/hal_dumas_metadata_cleaned.json",
    figures_dir="figures",
    models_dir="models",
    num_topics=5,
    pca_dims=50
):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    texts, tokens = load_corpus(input_path)
    n_docs = len(texts)
    print(f"Documents chargés : {n_docs}")
    if n_docs < 3:
        print("Pas assez de documents pour clustering.")
        return

    stop_words = load_stopwords()
    X, vectorizer = build_tfidf(texts, stop_words)
    joblib.dump(vectorizer, os.path.join(models_dir, "tfidf_vectorizer.joblib"))

    X_dense = X.toarray()
    if X_dense.shape[1] > pca_dims:
        pca = PCA(n_components=pca_dims, random_state=42)
        X_red = pca.fit_transform(X_dense)
        joblib.dump(pca, os.path.join(models_dir, "pca.joblib"))
    else:
        X_red = X_dense

    k_min = 2
    k_max = min(10, n_docs - 1)
    k_values = list(range(k_min, k_max + 1))
    inertias = []
    silhouettes = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = km.fit_predict(X_red)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_red, labels))

    plot_elbow(k_values, inertias, os.path.join(figures_dir, "elbow.png"))
    plot_silhouette(k_values, silhouettes, os.path.join(figures_dir, "silhouette.png"))

    best_k = k_values[int(np.argmax(silhouettes))]
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_red)
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans.joblib"))

    perplexity = max(2, min(30, (n_docs - 1) // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
    tsne_coords = tsne.fit_transform(X_red)
    plot_tsne(tsne_coords, labels, os.path.join(figures_dir, "tsne.png"))

    all_tokens = [t for doc in tokens for t in doc if t not in stop_words]
    wordcloud = WordCloud(width=900, height=450, background_color="white").generate(" ".join(all_tokens))
    wordcloud.to_file(os.path.join(figures_dir, "wordcloud.png"))

    tokenized_docs = [[t for t in doc if t not in stop_words and len(t) > 2] for doc in tokens]
    dictionary = corpora.Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.9)
    corpus = [dictionary.doc2bow(text) for text in tokenized_docs]
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
    lda_model.save(os.path.join(models_dir, "lda_model"))
    dictionary.save(os.path.join(models_dir, "lda_dictionary"))

    with open(os.path.join(figures_dir, "lda_topics.txt"), "w", encoding="utf-8") as f:
        for idx, topic in lda_model.print_topics(num_words=8):
            f.write(f"Thème {idx + 1}: {topic}\n")

    plot_lda_topics(lda_model, os.path.join(figures_dir, "lda_topics.png"))

    cluster_counts = Counter(labels)
    with open(os.path.join(figures_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cluster_counts.items()}, f, ensure_ascii=False, indent=2)

    np.savetxt(os.path.join(figures_dir, "tsne_coords.csv"), tsne_coords, delimiter=",")

    print(f"Meilleur k (silhouette) : {best_k}")
    print(f"Wordcloud -> {figures_dir}/wordcloud.png")
    print(f"LDA -> {figures_dir}/lda_topics.txt, {figures_dir}/lda_topics.png")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline complet Module 2.")
    parser.add_argument("--input", default="data/cleaned/hal_dumas_metadata_cleaned.json")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--models", default="models")
    parser.add_argument("--topics", type=int, default=5)
    parser.add_argument("--pca", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        figures_dir=args.figures,
        models_dir=args.models,
        num_topics=args.topics,
        pca_dims=args.pca
    )
