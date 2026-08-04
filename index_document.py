from ingest import extract_text_from_pdf
from chunking import chunk_text
from vectorstore import store_chunks
from bm25 import build_bm25


def index_document(pdf_path):

    with open(pdf_path, "rb") as file:
        text = extract_text_from_pdf(file)

    chunks = chunk_text(text)

    store_chunks(chunks)

    build_bm25(chunks)

    print("✅ Document indexed successfully")


if __name__ == "__main__":
    index_document("data/evaluation.pdf")