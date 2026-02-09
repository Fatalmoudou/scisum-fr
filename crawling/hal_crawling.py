import os
import json
import requests

HAL_API_URL = "https://api.archives-ouvertes.fr/search/"

def crawl_hal(query, max_results=120, batch_size=100, output_dir="data/raw"):
    """
    Télécharge des articles HAL en français à partir d'un mot-clé.

    :param query: mot-clé de recherche
    :param max_results: nombre maximum d'articles
    :param output_dir: dossier de sortie
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata = []
    start = 0
    while len(metadata) < max_results:
        rows = min(batch_size, max_results - len(metadata))
        params = {
            "q": query,
            "rows": rows,
            "start": start,
            "fl": "title_s,authFullName_s,producedDateY_i,abstract_s,language_s,uri_s,files_s",
            "fq": "language_s:fr",
            "wt": "json"
        }

        response = requests.get(HAL_API_URL, params=params)
        response.raise_for_status()

        data = response.json()
        documents = data.get("response", {}).get("docs", [])
        if not documents:
            break

        for doc in documents:
            title = doc.get("title_s", [""])
            abstract = doc.get("abstract_s", [""])
            entry = {
                "title": title[0] if isinstance(title, list) and title else title,
                "authors": doc.get("authFullName_s", []),
                "year": doc.get("producedDateY_i"),
                "abstract": abstract[0] if isinstance(abstract, list) and abstract else abstract,
                "uri": doc.get("uri_s")
            }
            metadata.append(entry)

        start += rows

    # Sauvegarde des métadonnées
    meta_path = os.path.join(output_dir, "hal_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"{len(metadata)} articles récupérés.")
    print(f"Métadonnées sauvegardées dans {meta_path}")


if __name__ == "__main__":
    crawl_hal("apprentissage automatique", max_results=120)
