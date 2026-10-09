import json
import time
from src.ingest import ingest_all
from src.embed_store import build_index
from src.retrieve import retrieve

with open("eval/questions.json", encoding="utf-8") as f:
    QUESTIONS = json.load(f)


def hit_rate(k):
    hits = 0
    for q in QUESTIONS:
        chunks = retrieve(q["question"], k)
        text = " ".join(" ".join(c["text"].lower().split()) for c in chunks)
        if any(kw.lower() in text for kw in q["keywords"]):
            hits += 1
    return hits / len(QUESTIONS)


if __name__ == "__main__":
    print(f"{len(QUESTIONS)} questions\n")
    print("chunk_size overlap  k  hit_rate")
    for size, overlap in [(400, 80), (800, 150), (1200, 200)]:
        recs = ingest_all(size=size, overlap=overlap)
        build_index(recs, reset=True)
        for k in [2, 4, 6]:
            t = time.time()
            hr = hit_rate(k)
            print(f"{size:>10} {overlap:>7} {k:>2}  {hr:.0%}   ({time.time()-t:.1f}s)")