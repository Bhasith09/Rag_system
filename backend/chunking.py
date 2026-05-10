#chunking.py


from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(paragraphs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=150
    )
    chunks=[]

    for para_data in paragraphs:
        split_chunks=splitter.split_text(para_data["text"])

        for chunk in split_chunks:
            chunks.append({
                "text": chunk,
                "page": para_data["page"],
                "paragraph": para_data["paragraph"],
                "source": para_data["source"]
            })
    return chunks

