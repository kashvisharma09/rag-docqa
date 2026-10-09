from src.config import TOP_K
from src.embed_store import get_model, get_collection


def retrieve(query, k=TOP_K):
    model = get_model()
    col = get_collection()
    q_emb = model.encode([query]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    hits = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "score": round(1 - dist, 3),
        })
    return hits


if __name__ == "__main__":
    q = "How is Parkinsonian tremor detected from IMU data?"
    for h in retrieve(q):
        print(f"[{h['source']} p{h['page']}] score={h['score']}")
        print(h["text"][:200].replace("\n", " "), "\n")