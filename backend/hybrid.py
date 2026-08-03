# hybrid.py

from backend.vectorstore import search as vector_search
from backend.bm25 import bm25_search
from backend.reranker import rerank


def hybrid_search(query, k=5):
    vector_results = vector_search(query, k)
    bm25_results = bm25_search(query, k)

    # Merge results
    combined = vector_results + bm25_results

    # Remove duplicate chunks
    unique_docs = []
    seen = set()

    for doc in combined:
        key = (
            doc.page_content,
            doc.metadata.get("source"),
            doc.metadata.get("page")
        )

        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    # Rerank merged results
    final_docs = rerank(query, unique_docs, top_k=k)

    return final_docs