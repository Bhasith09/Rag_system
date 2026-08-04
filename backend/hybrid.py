from backend.vectorstore import search as vector_search
from backend.bm25 import bm25_search
from backend.reranker import rerank


def hybrid_search(query, k=5):
    vector_results = vector_search(query, k)
    bm25_results = bm25_search(query, k)

    # Merge results
    combined = vector_results + bm25_results

    # Remove duplicates based on chunk text
    seen = set()
    unique_docs = []

    for doc in combined:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    # Rerank the unique documents
    final_docs = rerank(query, unique_docs, top_k=k)

    return final_docs