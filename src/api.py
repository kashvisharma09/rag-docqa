from fastapi import FastAPI
from pydantic import BaseModel
from src.generate import answer

app = FastAPI(title="RAG Doc QA")


class Question(BaseModel):
    query: str
    k: int = 4


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    return answer(q.query, q.k)