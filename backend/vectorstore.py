from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = None

def store_chunks(chunks):
    """Create a FAISS vector database."""
    global vectorstore
    vectorstore = FAISS.from_documents(chunks, embeddings)


def search(query, k=5):
    """Search the vector database."""
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)


def reset_db():
    """Clear the current vector database."""
    global vectorstore
    vectorstore = None