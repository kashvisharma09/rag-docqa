import chromadb
from sentence_transformers import SentenceTransformer
from src.config import EMBED_MODEL, CHROMA_PATH, COLLECTION
from src.ingest import ingest_all

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def get_collection(reset=False):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    return client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"}
    )


def build_index(records, reset=True):
    col = get_collection(reset=reset)
    model = get_model()
    batch = 64
    for i in range(0, len(records), batch):
        part = records[i:i + batch]
        embeddings = model.encode([r["text"] for r in part]).tolist()
        col.add(
            ids=[r["id"] for r in part],
            documents=[r["text"] for r in part],
            embeddings=embeddings,
            metadatas=[{"source": r["source"], "page": r["page"]} for r in part],
        )
    return col


if __name__ == "__main__":
    recs = ingest_all()
    col = build_index(recs)
    print(f"Indexed {col.count()} chunks")