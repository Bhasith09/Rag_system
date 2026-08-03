# bm25.py

from rank_bm25 import BM25Okapi

bm25 = None
corpus = []
stored_chunks = []


def build_bm25(chunks):
    global bm25, corpus, stored_chunks
    stored_chunks = chunks
    corpus = [
        chunk.page_content.split()
        for chunk in chunks ]
    
    bm25 = BM25Okapi(corpus)


def bm25_search(query, k=5):
    global bm25, stored_chunks

    if bm25 is None:
        return []

    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)
    
    paired_results = zip(stored_chunks, scores)

    ranked_results = sorted(
        paired_results,
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked_results[:k]]