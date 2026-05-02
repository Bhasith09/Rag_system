from backend.vectorstore import search as vector_search
from backend.bm25 import bm25_search
from backend.reranker import rerank

def hybrid_search(query, k=5):
    vector_results=vector_search(query,k)
    bm25_result=bm25_search(query,k)

    #merge the results

    combined=vector_results+bm25_result

    #remove_dupliactes
    unique_docs=list(set(combined))

    #rerank the unique docs

    final_docs=rerank(query,unique_docs,top_k=k)

    return final_docs