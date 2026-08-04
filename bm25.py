# bm25.py

from rank_bm25 import BM25Okapi

bm25 = None
documents = []
tokenized_corpus = []


def build_bm25(chunks):
    global bm25, documents, tokenized_corpus

    documents = chunks
    tokenized_corpus = [doc.page_content.split() for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)


def bm25_search(query, k=5):
    global bm25, documents

    if bm25 is None:
        return []

    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:k]]