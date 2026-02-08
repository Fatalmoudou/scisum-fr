import json
import sqlite3
import os

INPUT_JSON = "data/cleaned/hal_metadata_cleaned.json"
DB_PATH = "data/db/corpus.db"

def create_db():
    """Créer la base SQLite et la table documents"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            authors TEXT,
            year INTEGER,
            uri TEXT,
            abstract_clean TEXT
        )
    """)
    conn.commit()
    conn.close()
    print(f"Base créée ou existante : {DB_PATH}")

def insert_documents():
    """Lire le JSON et insérer les documents dans la base"""
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        documents = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    for doc in documents:
        # authors en texte JSON pour garder la structure
        authors_text = json.dumps(doc.get("authors", []), ensure_ascii=False)
        c.execute("""
            INSERT INTO documents (title, authors, year, uri, abstract_clean)
            VALUES (?, ?, ?, ?, ?)
        """, (
            doc.get("title"),
            authors_text,
            doc.get("year"),
            doc.get("uri"),
            doc.get("abstract_clean")
        ))

    conn.commit()
    conn.close()
    print(f"{len(documents)} documents insérés dans la base.")

if __name__ == "__main__":
    create_db()
    insert_documents()
