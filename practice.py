from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile

def extract_text_from_pdf(file)->str :
    with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as temp_file:
        temp_file.write(file.file.read())
        temp_path=temp_file.name
        
    try:
        loader=PyPDFLoader(temp_path)
        pages=loader.load()
        
        text=""
        for page in pages:
            text=text+page.page_content + "\n"
            
        return text
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        



