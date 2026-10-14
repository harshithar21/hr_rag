import hashlib
import json
from pathlib import Path
import fitz  # PyMuPDF
import docx
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PayloadSchemaType, Filter, FieldCondition, MatchValue

# Configuration
DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"
QDRANT_PATH = Path(__file__).resolve().parents[2] / "data" / "qdrant"
COLLECTION_NAME = "documents"
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters

def get_text_from_pdf(path: Path):
    doc = fitz.open(path)
    texts = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        meta = {"source": str(path.name), "page": page_num + 1}
        texts.append((text, meta))
    return texts

def get_text_from_docx(path: Path):
    doc = docx.Document(path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    # docx has no explicit pages; treat whole doc as single source
    return [("\n".join(full_text), {"source": str(path.name), "page": None})]

def get_text_from_txt(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [(f.read(), {"source": str(path.name), "page": None})]

def chunk_text(text: str, meta: dict):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]
        chunk_meta = meta.copy()
        chunk_meta.update({"chunk_index": len(chunks), "offset": start})
        chunks.append((chunk, chunk_meta))
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
    return chunks

def embed_text(text: str):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    # Ollama returns a list of embeddings under 'embeddings'
    return data["embeddings"][0]

def get_or_create_collection(client: QdrantClient, vector_size: int):
    if COLLECTION_NAME not in client.get_collections().collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    else:
        # Ensure vector size matches
        coll_info = client.get_collection(COLLECTION_NAME)
        if coll_info.config.params.vectors.size != vector_size:
            raise ValueError("Existing collection vector size does not match embedding size")

def hash_chunk(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def index_documents():
    client = QdrantClient(path=str(QDRANT_PATH))
    # First, collect all chunks and compute embeddings to know vector size
    all_chunks = []
    for file_path in DOCUMENTS_DIR.rglob("*.*"):
        if file_path.suffix.lower() == ".pdf":
            sources = get_text_from_pdf(file_path)
        elif file_path.suffix.lower() == ".docx":
            sources = get_text_from_docx(file_path)
        elif file_path.suffix.lower() == ".txt":
            sources = get_text_from_txt(file_path)
        else:
            continue
        for text, meta in sources:
            for chunk_text, chunk_meta in chunk_text(text, meta):
                chunk_id = hash_chunk(chunk_text)
                all_chunks.append((chunk_id, chunk_text, chunk_meta))
    if not all_chunks:
        print("No documents found to index.")
        return
    # Compute embeddings (first one to get size)
    sample_embedding = embed_text(all_chunks[0][1])
    vector_size = len(sample_embedding)
    get_or_create_collection(client, vector_size)
    # Upsert chunks, avoiding duplicates
    for chunk_id, chunk_text, chunk_meta in all_chunks:
        # Check if point already exists
        try:
            existing = client.scroll(COLLECTION_NAME, limit=1, filter=Filter(must=[
                FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id))
            ]))
            if existing[0]:  # points returned
                continue
        except Exception:
            pass
        vector = embed_text(chunk_text)
        payload = {
            "text": chunk_text,
            "source": chunk_meta.get("source"),
            "page": chunk_meta.get("page"),
            "chunk_index": chunk_meta.get("chunk_index"),
            "offset": chunk_meta.get("offset"),
            "chunk_id": chunk_id,
        }
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                {
                    "id": chunk_id,
                    "vector": vector,
                    "payload": payload,
                }
            ],
        )
    print(f"Indexed {len(all_chunks)} chunks into collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    index_documents()
