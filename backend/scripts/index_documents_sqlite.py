import sqlite3
import json
from pathlib import Path
import pymupdf as fitz  # PyMuPDF (updated import)
import docx
import requests
import hashlib
from sentence_transformers import CrossEncoder
import numpy as np
import sqlite_vec

# ---------------------------------------------------------------------------
# Configuration (same as previous script)
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = BASE_DIR / "data" / "documents"
DB_PATH = BASE_DIR / "data" / "search.db"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------------------
# Helper functions for text extraction
# ---------------------------------------------------------------------------
def get_text_from_pdf(path: Path):
    doc = fitz.open(path)
    texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        texts.append((page.get_text(), {"source": path.name, "page": page_num + 1}))
    return texts

def get_text_from_docx(path: Path):
    doc = docx.Document(path)
    full = "\n".join(p.text for p in doc.paragraphs)
    return [(full, {"source": path.name, "page": None})]

def get_text_from_txt(path: Path):
    return [(path.read_text(encoding="utf-8"), {"source": path.name, "page": None})]

def chunk_text(text: str, meta: dict):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]
        md = meta.copy()
        md.update({"chunk_index": len(chunks), "offset": start})
        chunks.append((chunk, md))
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks

def embed_text(text: str) -> list[float]:
    r = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    # Ollama returns {"embedding": [...]} (single vector)
    return data["embedding"]

def hash_chunk(txt: str) -> str:
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# SQLite schema creation
# ---------------------------------------------------------------------------
def init_db(conn: sqlite3.Connection):
    # Enable the sqlite-vec extension
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    cur = conn.cursor()
    # Main table
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            source TEXT,
            page INTEGER,
            chunk_index INTEGER,
            offset INTEGER,
            chunk_id TEXT UNIQUE,
            embedding VECTOR
        )
        """
    )
    # Full‑text virtual table (FTS5) linked to main table via rowid
    cur.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            source,
            content='chunks',
            content_rowid='rowid'
        )
        """
    )
    # Triggers to keep FTS in sync
    cur.executescript(
        """
        CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, source) VALUES (new.rowid, new.text, new.source);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, source) VALUES('delete', old.rowid, old.text, old.source);
        END;
        CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, source) VALUES('delete', old.rowid, old.text, old.source);
            INSERT INTO chunks_fts(rowid, text, source) VALUES (new.rowid, new.text, new.source);
        END;
        """
    )
    conn.commit()

# ---------------------------------------------------------------------------
# Indexing function
# ---------------------------------------------------------------------------
def index_documents():
    conn = sqlite3.connect(str(DB_PATH))
    init_db(conn)
    cur = conn.cursor()
    total_indexed = 0
    for file_path in DOCUMENTS_DIR.rglob("*.*"):
        if file_path.suffix.lower() == ".pdf":
            sources = get_text_from_pdf(file_path)
        elif file_path.suffix.lower() == ".docx":
            sources = get_text_from_docx(file_path)
        elif file_path.suffix.lower() == ".txt":
            sources = get_text_from_txt(file_path)
        else:
            continue
        for txt, meta in sources:
            for chunk_txt, chunk_meta in chunk_text(txt, meta):
                chunk_id = hash_chunk(chunk_txt)
                # duplicate‑check
                cur.execute("SELECT 1 FROM chunks WHERE chunk_id = ?", (chunk_id,))
                if cur.fetchone():
                    continue
                vec = embed_text(chunk_txt)
                # SQLite expects a binary blob for VECTOR; sqlite_vec handles conversion
                cur.execute(
                    "INSERT INTO chunks (id, text, source, page, chunk_index, offset, chunk_id, embedding) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk_id,
                        chunk_txt,
                        chunk_meta.get("source"),
                        chunk_meta.get("page"),
                        chunk_meta.get("chunk_index"),
                        chunk_meta.get("offset"),
                        chunk_id,
                        json.dumps(vec),  # vector stored as JSON string; sqlite-vec will parse it
                    ),
                )
                total_indexed += 1
    conn.commit()
    conn.close()
    print(f"Indexed {total_indexed} chunks into SQLite DB at {DB_PATH}")

if __name__ == "__main__":
    index_documents()
