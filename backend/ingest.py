import PyPDF2


def extract_text_from_pdf(file):

    reader = PyPDF2.PdfReader(file)

    pages = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text() or ""

        # Split into paragraphs
        paragraphs = text.split("\n\n")

        for para_num, para in enumerate(paragraphs):

            para = para.strip()

            if para:

                pages.append({
                    "text": para,
                    "page": page_num + 1,
                    "paragraph": para_num + 1,
                    "source": file.name
                })

    return pages