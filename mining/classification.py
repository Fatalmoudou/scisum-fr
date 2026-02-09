import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

STOPWORDS_PATH = Path("analysis/stopwords_fr.txt")


def load_stopwords():
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texts = []
    labels = []
    for doc in data:
        text = doc.get("texte_nettoye") or doc.get("abstract_clean") or ""
        if not text:
            continue
        label = infer_label(doc)
        if label is None:
            continue
        texts.append(text)
        labels.append(label)
    return texts, labels


def _normalize_keywords(values):
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    return [v.lower() for v in values if isinstance(v, str)]


def infer_label(doc):
    # Weak labels based on keywords / domain.
    kw = _normalize_keywords(doc.get("keyword_s") or doc.get("domaine"))
    if not kw:
        return None

    joined = " ".join(kw)

    def has_any(terms):
        return any(t in joined for t in terms)

    if has_any(["traitement automatique", "tal", "langage", "linguistique", "nlp"]):
        return "TAL"
    if has_any(["fouille", "data mining", "mining", "exploration"]):
        return "FOUILLE"
    if has_any(["apprentissage", "machine", "deep", "réseau", "reseau", "classification", "svm"]):
        return "IA"
    return None


def train_models(texts, labels, out_dir="models", fig_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    stop_words = load_stopwords()
    vectorizer = TfidfVectorizer(stop_words=list(stop_words), min_df=2, max_df=0.9)
    X = vectorizer.fit_transform(texts)
    y = labels

    label_counts = Counter(y)
    if len(label_counts) < 2:
        raise ValueError("Pas assez de classes pour classifier.")

    # Split
    stratify = y if min(label_counts.values()) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # SVM
    svm = LinearSVC()
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)

    # RandomForest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Reports
    svm_report = classification_report(y_test, svm_pred, digits=3)
    rf_report = classification_report(y_test, rf_pred, digits=3)

    with open(os.path.join(fig_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("=== LinearSVC ===\n")
        f.write(svm_report + "\n")
        f.write("=== RandomForest ===\n")
        f.write(rf_report + "\n")

    # Confusion matrices
    ConfusionMatrixDisplay.from_predictions(y_test, svm_pred)
    plt.title("Confusion Matrix - LinearSVC")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_svm.png"))
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
    plt.title("Confusion Matrix - RandomForest")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "confusion_rf.png"))
    plt.close()

    # Save models
    joblib.dump(vectorizer, os.path.join(out_dir, "clf_tfidf_vectorizer.joblib"))
    joblib.dump(svm, os.path.join(out_dir, "clf_svm.joblib"))
    joblib.dump(rf, os.path.join(out_dir, "clf_rf.joblib"))

    return label_counts


def parse_args():
    parser = argparse.ArgumentParser(description="Classification supervisée (labels faibles).")
    parser.add_argument("--input", default="data/cleaned/hal_dumas_metadata_cleaned.json")
    parser.add_argument("--models", default="models")
    parser.add_argument("--figures", default="figures")
    return parser.parse_args()


def main():
    args = parse_args()
    texts, labels = load_corpus(args.input)
    counts = Counter(labels)
    print("Répartition des labels:", dict(counts))
    train_models(texts, labels, out_dir=args.models, fig_dir=args.figures)
    print("Classification terminée.")


if __name__ == "__main__":
    main()
