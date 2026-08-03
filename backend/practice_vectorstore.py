from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


def extract_text_from_pdf(file_path):
    loaders=PyPDFLoader(file_path)
    pages=loaders.load()
    
    paragraph=[]
    
    for page in pages:
        page_text=page.page_content.strip()
        
        split_paragraph=[]
        for i in page_text.split("\n\n"):
            i=i.strip()
            if i:
                split_paragraph.append(i)
            
            for para_num, para in enumerate(split_paragraph,start=1):
                paragraph.append(Document(page_content=para,
                    metadata={"source":page.metadata['source'],
                               "Page": page.metadata['page'],
                               'pargraph':para_num}
                ))
    return paragraph
                