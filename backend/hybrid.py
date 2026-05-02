from backend.vectorstore import search as vector_search
from backend.bm25 import bm25_search

def hybrid_search(query, k=5):
    vector_results=vector_search(query,k)
    bm25_result=bm25_search(query,k)

    #merge the results

    combined=vector_results+bm25_result

    #remove_dupliactes

    seen=set()
    unique_results=[]

    for doc in combined:
        if doc not in seen:
            unique_results.append(doc)
            seen.add(doc)

    return unique_results[:k]