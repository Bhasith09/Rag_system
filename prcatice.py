import io
from PyPDF2 import PdfReader

def extract_text_from_pdf(file):
    pdf_bytes=file.read()
    reader=PdfReader(io.BytesIO(pdf_bytes))
    text=""
    for i in reader.pages:
        page_text=i.extract_text()
        if page_text:
            text=text


