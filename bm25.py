# bm25.py

from rank_bm25 import BM25Okapi

bm25=None
corpus=[]

def build_bm25(chunks):
    global bm25,corpus

    corpus=[chunk.split() for chunk in chunks]
    bm25=BM25Okapi(corpus)

def bm25_search(query, k=5):
    global bm25, corpus
    if bm25 is None:
        return []
    
    tokenized_query=query.split()
    scores=bm25.get_scores(tokenized_query)

    ranked=sorted(
        list(zip(corpus, scores)),
        key=lambda x:x[1],
        reverse=True
    )

    return [" ".join(doc) for doc, _ in ranked[:k]]