import json
import os
import sqlite3
import hashlib
import re
import argparse

DEFAULT_INPUT_JSON = "data/cleaned/hal_dumas_metadata_cleaned.json"
DB_PATH = "data/db/corpus.db"

_RE_TOKEN = re.compile(r"[a-zàâçéèêëîïôûùüÿñæœ]+", re.IGNORECASE)
_RE_SENTENCE = re.compile(r"[.!?]+")


def _checksum(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _doc_stats(text):
    tokens = _RE_TOKEN.findall(text.lower())
    vocab = len(set(tokens))
    nb_mots = len(tokens)
    nb_phrases = len([s for s in _RE_SENTENCE.split(text) if s.strip()])
    return nb_mots, nb_phrases, vocab


def create_db(reset=False):
    """Créer la base SQLite et les tables."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if reset:
        c.execute("DROP TABLE IF EXISTS statistiques")
        c.execute("DROP TABLE IF EXISTS documents")
    c.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            titre TEXT,
            auteurs TEXT,
            date INTEGER,
            domaine TEXT,
            texte_nettoye TEXT,
            checksum TEXT UNIQUE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS statistiques (
            document_id INTEGER,
            nb_mots INTEGER,
            nb_phrases INTEGER,
            vocabulaire_size INTEGER,
            FOREIGN KEY(document_id) REFERENCES documents(id)
        )
    """)
    conn.commit()
    conn.close()
    print(f"Base créée ou existante : {DB_PATH}")


def insert_documents(input_json):
    """Lire le JSON et insérer les documents dans la base."""
    with open(input_json, "r", encoding="utf-8") as f:
        documents = json.load(f)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    inserted = 0
    for doc in documents:
        texte = doc.get("texte_nettoye") or doc.get("abstract_clean") or ""
        if not texte:
            continue
        checksum = _checksum(texte)
        auteurs = doc.get("authors") or doc.get("authFullName_s") or []
        auteurs_text = json.dumps(auteurs, ensure_ascii=False)
        titre = doc.get("title") or doc.get("title_s")
        if isinstance(titre, list):
            titre = titre[0] if titre else None
        domaine = doc.get("domaine") or doc.get("keyword_s")
        if isinstance(domaine, (list, tuple)):
            domaine = json.dumps(domaine, ensure_ascii=False)

        try:
            c.execute("""
                INSERT INTO documents (titre, auteurs, date, domaine, texte_nettoye, checksum)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                titre,
                auteurs_text,
                doc.get("year") or doc.get("producedDateY_i"),
                domaine,
                texte,
                checksum
            ))
        except sqlite3.IntegrityError:
            continue

        doc_id = c.lastrowid
        nb_mots, nb_phrases, vocab = _doc_stats(texte)
        c.execute("""
            INSERT INTO statistiques (document_id, nb_mots, nb_phrases, vocabulaire_size)
            VALUES (?, ?, ?, ?)
        """, (doc_id, nb_mots, nb_phrases, vocab))
        inserted += 1

    conn.commit()
    conn.close()
    print(f"{inserted} documents insérés dans la base.")


def parse_args():
    parser = argparse.ArgumentParser(description="Stocker le corpus nettoyé dans SQLite.")
    parser.add_argument("--input", default=DEFAULT_INPUT_JSON, help="Chemin du JSON nettoyé.")
    parser.add_argument("--reset", action="store_true", help="Réinitialiser la base (DROP tables).")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    create_db(reset=args.reset)
    insert_documents(args.input)
