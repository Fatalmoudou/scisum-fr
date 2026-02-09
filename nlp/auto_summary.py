import argparse
import json
import os
import re
from pathlib import Path

STOPWORDS_PATH = Path("analysis/stopwords_fr.txt")

_RE_TOKEN = re.compile(r"[a-zàâçéèêëîïôûùüÿñæœ]+", re.IGNORECASE)
_RE_SENT = re.compile(r"(?<=[.!?])\s+")


def load_stopwords():
    with open(STOPWORDS_PATH, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def sentence_tokenize(text):
    return [s.strip() for s in _RE_SENT.split(text) if s.strip()]


def tokenize(text):
    return [t.lower() for t in _RE_TOKEN.findall(text)]


def summarize(text, stop_words, ratio=0.2, min_sent=3, max_sent=7):
    sentences = sentence_tokenize(text)
    if not sentences:
        return ""
    tokens = tokenize(text)
    freq = {}
    for t in tokens:
        if t in stop_words:
            continue
        freq[t] = freq.get(t, 0) + 1

    scores = []
    for idx, sent in enumerate(sentences):
        sent_tokens = tokenize(sent)
        score = sum(freq.get(t, 0) for t in sent_tokens)
        scores.append((idx, score))

    k = max(min_sent, int(len(sentences) * ratio))
    k = min(k, max_sent, len(sentences))
    top_idx = {idx for idx, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:k]}
    selected = [sentences[i] for i in range(len(sentences)) if i in top_idx]
    return " ".join(selected)


def run(input_path, out_dir="summaries", ratio=0.2):
    os.makedirs(out_dir, exist_ok=True)
    stop_words = load_stopwords()
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    summaries = []
    for idx, doc in enumerate(data, start=1):
        text = doc.get("texte_nettoye") or doc.get("abstract_clean") or ""
        if not text:
            continue
        summ = summarize(text, stop_words, ratio=ratio)
        summaries.append({
            "id": idx,
            "title": doc.get("title") or doc.get("title_s"),
            "summary": summ
        })
        with open(os.path.join(out_dir, f"summary_{idx:04d}.txt"), "w", encoding="utf-8") as f:
            f.write(summ)

    with open(os.path.join(out_dir, "summaries.json"), "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    print(f"Summaries: {len(summaries)} -> {out_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description="Résumé automatique extractif.")
    parser.add_argument("--input", default="data/cleaned/hal_dumas_metadata_cleaned.json")
    parser.add_argument("--out", default="summaries")
    parser.add_argument("--ratio", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, out_dir=args.out, ratio=args.ratio)
