from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Add paragraph number within each page
    paragraph_count = {}

    for chunk in chunks:
        page = chunk.metadata.get("page", 0)

        paragraph_count[page] = paragraph_count.get(page, 0) + 1

        chunk.metadata["paragraph"] = paragraph_count[page]

    return chunks