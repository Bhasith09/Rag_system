from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_docs"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=DB_PATH,
    embedding_function=embeddings
)


def store_chunks(chunks):
    texts = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata)  # Preserve page & paragraph metadata
        metadata["chunk_index"] = i

        texts.append(chunk.page_content)
        metadatas.append(metadata)

    vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas
    )


def search(query, k=5):
    return vectorstore.similarity_search(query, k=k)


def reset_db():
    global vectorstore

    try:
        vectorstore.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )