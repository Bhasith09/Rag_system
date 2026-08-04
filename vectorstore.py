from sentence_transformers import SentenceTransformer
import chromadb
import uuid
import os

model = SentenceTransformer("all-MiniLM-L6-v2")

DB_PATH = "./chroma_db"

def get_client():
    # Always persistent for dev + CI safety
    return chromadb.PersistentClient(path=DB_PATH)

client = get_client()
collection = client.get_or_create_collection(name="rag_docs")


def store_chunks(chunks):
    embeddings = model.encode(chunks)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            embeddings=[embeddings[i].tolist()],
            ids=[str(uuid.uuid4())],
            metadatas=[{
                "chunk_index": i,
                "source": "pdf"
            }],
        )


def search(query, k=5):
    query_embedding = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    return results.get("documents", [[]])[0]


def reset_db():
    # SAFE RESET (no file deletion -> avoids WinError 32)
    try:
        collection.delete(where={})
    except Exception:
        pass