from sentence_transformers import CrossEncoder

reranker_model = CrossEncoder(
    'cross-encoder/ms-marco-MiniLM-L-6-v2'
)


def rerank(query, documents, top_k=5):

    if not documents:
        return []

    cleaned_docs = []

    # Ensure every document is dictionary format
    for doc in documents:

        if isinstance(doc, str):

            cleaned_docs.append({
                "text": doc,
                "page": "N/A",
                "paragraph": "N/A",
                "source": "Unknown"
            })

        else:
            cleaned_docs.append(doc)

    # Create (query, text) pairs
    pairs = [
        (query, doc["text"])
        for doc in cleaned_docs
    ]

    # Predict relevance scores
    scores = reranker_model.predict(pairs)

    # Combine docs + scores
    scored_docs = list(zip(cleaned_docs, scores))

    # Sort descending
    ranked_docs = sorted(
        scored_docs,
        key=lambda x: x[1],
        reverse=True
    )

    # Return top documents
    return [doc for doc, _ in ranked_docs[:top_k]]