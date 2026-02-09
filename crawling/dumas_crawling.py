import requests
import json
from pathlib import Path

BASE_URL = "https://api.archives-ouvertes.fr/search/"
OUTPUT_FILE = Path("data/raw/dumas_metadata.json")


def fetch_dumas_documents(
    query="apprentissage automatique",
    max_results=80,
    batch_size=100
):
    """
    Récupère des mémoires DUMAS via l'API HAL
    """
    docs = []
    start = 0
    print("Requête DUMAS en cours...")
    while len(docs) < max_results:
        rows = min(batch_size, max_results - len(docs))
        params = {
            "q": query,
            "fq": "docType_s:MEM",
            "fl": (
                "halId_s,"
                "title_s,"
                "abstract_s,"
                "authFullName_s,"
                "producedDateY_i,"
                "keyword_s"
            ),
            "rows": rows,
            "start": start,
            "wt": "json"
        }

        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        batch = data.get("response", {}).get("docs", [])
        if not batch:
            break
        docs.extend(batch)
        start += rows

    print(f"{len(docs)} documents DUMAS récupérés")
    return docs


def save_documents(docs):
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

    print(f"Données sauvegardées dans {OUTPUT_FILE}")


def main():
    docs = fetch_dumas_documents(
        query="traitement automatique du langage",
        max_results=80
    )
    save_documents(docs)


if __name__ == "__main__":
    main()
