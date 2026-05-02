# reranker.py

from sentence_transformers import CrossEncoder

# Initialize the cross-encoder model

reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query,documents, top_k=5):
    if not documents:
        return []
    
    #create (query, doc) pairs
    pairs=[(query,doc) for doc in documents]

    #get scores
    scores=reranker_model.predict(pairs)

    #combine docs + scores
    scored_docs=list(zip(documents, scores))

    #sort by score (high to low)

    ranked_docs=sorted(scored_docs,key=lambda x:x[1], reverse=True)


    #returnonly docs

    return [doc for doc, _ in ranked_docs[:top_k]]