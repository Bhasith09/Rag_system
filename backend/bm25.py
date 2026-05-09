# bm25.py

from rank_bm25 import BM25Okapi
import re
bm25=None
corpus=[]

def tokenize(text):
    return re.findall(r'\w+', text.lower())  #consider only letter digits numbers and underscore and then convert them to lower case

def build_bm25(chunks):
    global bm25,corpus

    corpus=[tokenize(chunk) for chunk in chunks]
    bm25=BM25Okapi(corpus)  #create bm25 index which cotains tf and idf values for each term in the corpus

def bm25_search(query, k=5):
    global bm25, corpus
    if bm25 is None:
        return []
    
    tokenized_query=tokenize(query)
    scores=bm25.get_scores(tokenized_query)

    ranked=sorted(
        list(zip(corpus, scores)),
        key=lambda x:x[1],
        reverse=True
    )

    return [" ".join(doc) for doc, _ in ranked[:k]]