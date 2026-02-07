import json
import os
from cleaning import clean_abstract

INPUT = "data/raw/hal_metadata.json"
OUTPUT = "data/cleaned/hal_metadata_cleaned.json"

def clean_corpus():
    with open(INPUT, "r", encoding="utf-8") as f:
        documents = json.load(f)

    for doc in documents:
        doc["abstract_clean"] = clean_abstract(doc.get("abstract", ""))

    # Création du dossier de sortie si nécessaire
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"{len(documents)} documents nettoyés.")
    print(f"Fichier sauvegardé dans {OUTPUT}")

if __name__ == "__main__":
    clean_corpus()
