# reranker.py

from sentence_transformers import CrossEncoder

# Initialize the Cross Encoder model
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, documents, top_k=5):
    if not documents:
        return []

    # Create (query, document_text) pairs
    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    # Predict relevance scores
    scores = reranker_model.predict(pairs)

    # Combine documents with their scores
    scored_docs = list(zip(documents, scores))

    # Sort by score (highest first)
    ranked_docs = sorted(
        scored_docs,
        key=lambda item: item[1],
        reverse=True
    )

    # Return only the top documents
    return [
        doc
        for doc, _ in ranked_docs[:top_k]
    ]