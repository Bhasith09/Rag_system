from fastapi import UploadFile
from backend.ingest import extract_text_from_pdf
from backend.chunking import chunk_text
from backend.vectorstore import reset_db, store_chunks
from backend.bm25 import build_bm25


def index_document(pdf_path):
    reset_db()

    with open(pdf_path, "rb") as f:
        uploaded_file = UploadFile(
            filename="evaluation.pdf",
            file=f
        )

        documents = extract_text_from_pdf(uploaded_file)

    chunks = chunk_text(documents)

    store_chunks(chunks)

    build_bm25(chunks)

    print("✅ Document indexed successfully")


if __name__ == "__main__":
    index_document("data/evaluation.pdf")