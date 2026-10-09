import ollama
from src.retrieve import retrieve

MODEL = "llama3.2"

SYSTEM = (
    "You answer questions using ONLY the provided context. "
    "Cite sources inline like [file.pdf p3]. "
    "If the context does not contain the answer, say you don't know."
)


def build_context(hits):
    return "\n\n".join(
        f"[{h['source']} p{h['page']}]\n{h['text']}" for h in hits
    )


def answer(query, k=4):
    hits = retrieve(query, k)
    context = build_context(hits)
    res = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    return {"answer": res["message"]["content"], "sources": hits}


if __name__ == "__main__":
    out = answer("How is Parkinsonian tremor detected from IMU data?")
    print(out["answer"])
    print("\nSources:")
    for h in out["sources"]:
        print(f"- {h['source']} p{h['page']} (score {h['score']})")