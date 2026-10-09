from pathlib import Path
from pypdf import PdfReader
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, RAW_DIR


def load_pdf(path):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def ingest_all(raw_dir=RAW_DIR, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    records = []
    for pdf in Path(raw_dir).glob("*.pdf"):
        for page_num, text in load_pdf(pdf):
            for j, chunk in enumerate(chunk_text(text, size, overlap)):
                records.append({
                    "id": f"{pdf.stem}_p{page_num}_c{j}",
                    "text": chunk,
                    "source": pdf.name,
                    "page": page_num,
                })
    return records


if __name__ == "__main__":
    recs = ingest_all()
    print(f"{len(recs)} chunks")
    print(recs[0] if recs else "No PDFs found")