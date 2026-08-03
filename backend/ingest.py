# pdf_loader.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def extract_text_from_pdf(file_path):

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    paragraphs = []

    for page in pages:

        page_text = page.page_content.strip()

        # Split into paragraph
        
        split_paragraphs = []
        for para in page_text.split("\n\n"):
            para = para.strip()
            if para:
                split_paragraphs.append(para)
                
                
        for para_num, para in enumerate(split_paragraphs, start=1):

            paragraphs.append(
                Document(
                    page_content=para,
                    metadata={
                        "source": page.metadata["source"],
                        "page": page.metadata["page"],
                        "paragraph": para_num
                    }
                )
            )

    return paragraphs