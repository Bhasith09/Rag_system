# chunking.py

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(paragraphs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(paragraphs)

    return chunks