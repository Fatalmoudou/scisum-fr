import os
import json
import requests

HAL_API_URL = "https://api.archives-ouvertes.fr/search/"

def crawl_hal(query, max_results=20, output_dir="data/raw"):
    """
    Télécharge des articles HAL en français à partir d'un mot-clé.

    :param query: mot-clé de recherche
    :param max_results: nombre maximum d'articles
    :param output_dir: dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)

    params = {
        "q": query,
        "rows": max_results,
        "fl": "title_s,authFullName_s,producedDateY_i,abstract_s,language_s,uri_s,files_s",
        "fq": "language_s:fr",
        "wt": "json"
    }

    response = requests.get(HAL_API_URL, params=params)
    response.raise_for_status()

    data = response.json()
    documents = data.get("response", {}).get("docs", [])

    metadata = []

    for doc in documents:
        entry = {
            "title": doc.get("title_s", [""])[0],
            "authors": doc.get("authFullName_s", []),
            "year": doc.get("producedDateY_i"),
            "abstract": doc.get("abstract_s", [""])[0],
            "uri": doc.get("uri_s")
        }
        metadata.append(entry)

    # Sauvegarde des métadonnées
    meta_path = os.path.join(output_dir, "hal_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"{len(metadata)} articles récupérés.")
    print(f"Métadonnées sauvegardées dans {meta_path}")


if __name__ == "__main__":
    crawl_hal("apprentissage automatique", max_results=10)
