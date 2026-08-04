from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from backend.ingest import extract_text_from_pdf
from backend.chunking import chunk_text
from backend.vectorstore import store_chunks, reset_db
from backend.bm25 import build_bm25
from backend.hybrid import hybrid_search
from backend.context import build_context
from backend.llm import generate_answer

app = FastAPI(
    title="Production Hybrid RAG API",
    version="1.0.0"
)

# Keep chunks in memory for the current uploaded document
indexed = False


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "Production Hybrid RAG API Running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global indexed

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )

    reset_db()

    documents = extract_text_from_pdf(file)

    chunks = chunk_text(documents)

    store_chunks(chunks)

    build_bm25(chunks)

    indexed = True

    return {
        "message": "Document indexed successfully",
        "chunks": len(chunks)
    }


@app.post("/ask")
async def ask_question(request: QuestionRequest):
    global indexed

    if not indexed:
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF first."
        )

    docs = hybrid_search(request.question, k=5)

    if not docs:
        return {
            "answer": "I don't know based on the provided documents.",
            "contexts": []
        }

    context = build_context(docs)

    answer = generate_answer(
        request.question,
        context
    )

    contexts = []

    for doc in docs:
        contexts.append({
            "page": doc.metadata.get("page", 0) + 1,
            "paragraph": doc.metadata.get("paragraph", 1),
            "text": doc.page_content
        })

    return {
        "answer": answer,
        "contexts": contexts
    }