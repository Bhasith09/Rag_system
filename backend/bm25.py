# bm25.py

from rank_bm25 import BM25Okapi
import re
bm25=None
corpus=[]
stored_chunks=[]


def build_bm25(chunks):
    global bm25,corpus,stored_chunks

    stored_chunks=chunks
    corpus=[chunk["text"].split() for chunk in chunks]
    bm25=BM25Okapi(corpus)  #create bm25 index which cotains tf and idf values for each term in the corpus

def bm25_search(query, k=5):
    global bm25, corpus, stored_chunks
    if bm25 is None:
        return []
    
    tokenized_query=query.split()
    scores=bm25.get_scores(tokenized_query)

    ranked=sorted(
        list(zip(stored_chunks, scores)),
        key=lambda x:x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:k]]