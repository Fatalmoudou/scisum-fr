import argparse
import json
import os
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

_RE_TOKEN = re.compile(r"[a-zàâçéèêëîïôûùüÿñæœ]+", re.IGNORECASE)
STOPWORDS_PATH = Path("analysis/stopwords_fr.txt")


class SOM:
    def __init__(self, rows, cols, dim, seed=42):
        self.rows = rows
        self.cols = cols
        self.dim = dim
        rng = np.random.default_rng(seed)
        self.weights = rng.random((rows, cols, dim))

    def find_bmu(self, vector):
        diffs = self.weights - vector
        dists = np.linalg.norm(diffs, axis=2)
        bmu_idx = np.unravel_index(np.argmin(dists), dists.shape)
        return bmu_idx

    def neighborhood(self, bmu, sigma):
        rr, cc = np.indices((self.rows, self.cols))
        br, bc = bmu
        dist_sq = (rr - br) ** 2 + (cc - bc) ** 2
        return np.exp(-dist_sq / (2 * (sigma ** 2)))

    def train(self, data, epochs=200, lr=0.5, sigma=None):
        if sigma is None:
            sigma = max(self.rows, self.cols) / 2
        time_constant = epochs / np.log(sigma)

        for t in range(epochs):
            lr_t = lr * np.exp(-t / epochs)
            sigma_t = sigma * np.exp(-t / time_constant)
            for vector in data:
                bmu = self.find_bmu(vector)
                neigh = self.neighborhood(bmu, sigma_t)[:, :, np.newaxis]
                self.weights += lr_t * neigh * (vector - self.weights)

    def map_vectors(self, data):
        return [self.find_bmu(vec) for vec in data]

    def quantization_error(self, data):
        errors = []
        for vector in data:
            r, c = self.find_bmu(vector)
            errors.append(np.linalg.norm(vector - self.weights[r, c]))
        return float(np.mean(errors)) if errors else 0.0

    def u_matrix(self):
        umat = np.zeros((self.rows, self.cols))
        for r in range(self.rows):
            for c in range(self.cols):
                neighbors = []
                if r > 0:
                    neighbors.append(self.weights[r - 1, c])
                if r < self.rows - 1:
                    neighbors.append(self.weights[r + 1, c])
                if c > 0:
                    neighbors.append(self.weights[r, c - 1])
                if c < self.cols - 1:
                    neighbors.append(self.weights[r, c + 1])
                if neighbors:
                    dists = [np.linalg.norm(self.weights[r, c] - n) for n in neighbors]
                    umat[r, c] = np.mean(dists)
        return umat


def load_stopwords():
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = [
        doc.get("texte_nettoye") or doc.get("abstract_clean") or ""
        for doc in data
    ]
    texts = [t for t in texts if t]
    tokens = [_RE_TOKEN.findall(text.lower()) for text in texts]
    return texts, tokens


def build_tfidf(texts, stop_words, max_features=5000):
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        max_features=max_features
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def plot_umatrix(umat, out_path):
    plt.figure(figsize=(6, 6))
    plt.imshow(umat, cmap="viridis")
    plt.colorbar()
    plt.title("SOM U-Matrix")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_som_hits(mapped, rows, cols, out_path):
    hits = np.zeros((rows, cols))
    for r, c in mapped:
        hits[r, c] += 1
    plt.figure(figsize=(6, 6))
    plt.imshow(hits, cmap="magma")
    plt.colorbar()
    plt.title("SOM Hits")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def run(
    input_path="data/cleaned/hal_dumas_metadata_cleaned.json",
    rows=15,
    cols=15,
    epochs=200,
    lr=0.5,
    pca_dims=100,
    figures_dir="figures",
    models_dir="models"
):
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    texts, _ = load_corpus(input_path)
    stop_words = load_stopwords()
    X, vectorizer = build_tfidf(texts, stop_words)
    joblib.dump(vectorizer, os.path.join(models_dir, "som_tfidf_vectorizer.joblib"))

    X_dense = X.toarray()
    if X_dense.shape[1] > pca_dims:
        pca = PCA(n_components=pca_dims, random_state=42)
        X_red = pca.fit_transform(X_dense)
        joblib.dump(pca, os.path.join(models_dir, "som_pca.joblib"))
    else:
        X_red = X_dense

    som = SOM(rows=rows, cols=cols, dim=X_red.shape[1])
    som.train(X_red, epochs=epochs, lr=lr)

    mapped = som.map_vectors(X_red)
    qe = som.quantization_error(X_red)

    np.save(os.path.join(models_dir, "som_weights.npy"), som.weights)
    with open(os.path.join(models_dir, "som_bmu_coords.json"), "w", encoding="utf-8") as f:
        json.dump([{"r": r, "c": c} for r, c in mapped], f, ensure_ascii=False, indent=2)

    plot_umatrix(som.u_matrix(), os.path.join(figures_dir, "som_umatrix.png"))
    plot_som_hits(mapped, rows, cols, os.path.join(figures_dir, "som_hits.png"))

    # Comparison with KMeans
    k = min(8, max(2, len(texts) // 25))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X_red)
    sil = silhouette_score(X_red, labels) if len(set(labels)) > 1 else 0.0
    joblib.dump(kmeans, os.path.join(models_dir, "som_kmeans_baseline.joblib"))

    # Optional MiniSom comparison
    minisom_qe = None
    try:
        from minisom import MiniSom
        ms = MiniSom(rows, cols, X_red.shape[1], sigma=rows/2, learning_rate=lr, random_seed=42)
        ms.random_weights_init(X_red)
        ms.train_batch(X_red, epochs)
        minisom_qe = float(ms.quantization_error(X_red))
    except Exception:
        minisom_qe = None

    with open(os.path.join(figures_dir, "som_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Documents: {len(texts)}\n")
        f.write(f"SOM grid: {rows}x{cols}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Quantization error (SOM): {qe:.4f}\n")
        f.write(f"KMeans baseline k={k}, silhouette={sil:.4f}\n")
        if minisom_qe is not None:
            f.write(f"MiniSom quantization error: {minisom_qe:.4f}\n")
        else:
            f.write("MiniSom not available\n")

    print(f"SOM trained. QE={qe:.4f}")
    print(f"Saved: {figures_dir}/som_umatrix.png, {figures_dir}/som_hits.png")
    print(f"Report: {figures_dir}/som_report.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="SOM from scratch on text corpus.")
    parser.add_argument("--input", default="data/cleaned/hal_dumas_metadata_cleaned.json")
    parser.add_argument("--rows", type=int, default=15)
    parser.add_argument("--cols", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--pca", type=int, default=100)
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--models", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        input_path=args.input,
        rows=args.rows,
        cols=args.cols,
        epochs=args.epochs,
        lr=args.lr,
        pca_dims=args.pca,
        figures_dir=args.figures,
        models_dir=args.models
    )
