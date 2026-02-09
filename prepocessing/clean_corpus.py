import glob
import json
import os
from cleaning import clean_abstract

HAL_INPUT = "data/raw/hal_metadata.json"
DUMAS_INPUT = "data/raw/dumas_metadata.json"
OUTPUT_HAL = "data/cleaned/hal_metadata_cleaned.json"
OUTPUT_DUMAS = "data/cleaned/dumas_metadata_cleaned.json"
OUTPUT_MERGED = "data/cleaned/hal_dumas_metadata_cleaned.json"
CORPUS_DIR = "corpus"

def _load_docs(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_docs(documents):
    cleaned = 0
    for doc in documents:
        raw = doc.get("abstract") or doc.get("abstract_s") or ""
        if isinstance(raw, list):
            raw = raw[0] if raw else ""
        doc["texte_nettoye"] = clean_abstract(raw)
        if doc["texte_nettoye"]:
            cleaned += 1
    return cleaned


def _save_docs(documents, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)


def _write_corpus_files(documents, subdir, prefix):
    out_dir = os.path.join(CORPUS_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    written = 0
    for idx, doc in enumerate(documents, start=1):
        text = doc.get("texte_nettoye") or ""
        if not text:
            continue
        filename = f"{prefix}_{idx:04d}.txt"
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        written += 1
    return written
def clean_corpus():
    hal_docs = _load_docs(HAL_INPUT)
    dumas_docs = _load_docs(DUMAS_INPUT)

    cleaned_hal = _clean_docs(hal_docs)
    cleaned_dumas = _clean_docs(dumas_docs)

    _save_docs(hal_docs, OUTPUT_HAL)
    _save_docs(dumas_docs, OUTPUT_DUMAS)

    merged = hal_docs + dumas_docs
    _save_docs(merged, OUTPUT_MERGED)

    hal_written = _write_corpus_files(hal_docs, "hal", "hal")
    dumas_written = _write_corpus_files(dumas_docs, "dumas", "dumas")

    print(f"HAL: {cleaned_hal} documents nettoyés -> {OUTPUT_HAL}")
    print(f"DUMAS: {cleaned_dumas} documents nettoyés -> {OUTPUT_DUMAS}")
    print(f"Fusion: {len(merged)} documents -> {OUTPUT_MERGED}")
    print(f"Fichiers texte: {hal_written} (HAL), {dumas_written} (DUMAS) -> {CORPUS_DIR}/")

if __name__ == "__main__":
    clean_corpus()
