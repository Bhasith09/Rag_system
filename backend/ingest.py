from langchain_community.document_loaders import PyPDFLoader
import tempfile
import os


def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.file.read())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        return documents

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)