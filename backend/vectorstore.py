#vectorstore.py

from sentence_transformers import SentenceTransformer
import chromadb
import uuid

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.EphemeralClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_docs")


def store_chunks(chunks):
    embeddings = model.encode(chunks)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[str(uuid.uuid4())],
            metadatas=[{"chunk_index": i,
            "source": "pdf"}],
        )


def search(query, k=5):
    query_embedding = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results.get("documents", [[]])[0]


def reset_db():
    """Safely delete all stored vectors"""
    data = collection.get()
    if data and data.get("ids"):
        collection.delete(ids=data["ids"])