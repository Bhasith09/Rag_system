from rank_bm25 import BM25Okapi

bm25=None
stored_chunks=[]
corpus=[]

def build_bm25(chunks):
    
    global bm25, stored_chunks, corpus
    stored_chunks=chunks
    corpus=[]
    for chunk in chunks:
        corpus.append(chunk.page_content.split())
        
    bm25=BM25Okapi(corpus)