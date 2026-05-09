#vectorstore.py

from sentence_transformers import SentenceTransformer
import chromadb
import uuid

model = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_docs")


def store_chunks(chunks):
    texts=[chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk["text"]],
            embeddings=[embeddings[i].tolist()],
            ids=[str(uuid.uuid4())],

            metadatas=[{
                "page": chunk["page"],
                "paragraph": chunk["paragraph"],
                "source": chunk["source"]
            }],
        )


def search(query, k=5):
    query_embedding = model.encode([query])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    docs=results.get("documents", [[]])[0]
    metas=results.get("metadatas", [[]])[0]

    combined=[]
    for doc, meta in zip(docs, metas):
        combined.append({
            "text": doc,
            "page": meta["page"],
            "paragraph": meta["paragraph"],
            "source": meta["source"]
        })
    return combined



def reset_db():
    """Safely delete all stored vectors"""
    data = collection.get()
    if data and data.get("ids"):
        collection.delete(ids=data["ids"])