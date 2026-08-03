from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunkigs(paragraph):
    splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=150)
    chunks=splitter.split_documents(paragraph)
    
    return chunks