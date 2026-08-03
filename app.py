# streamlit_app.py

import os
import tempfile
import streamlit as st

from backend.ingest import extract_text_from_pdf
from backend.chunking import chunk_text
from backend.vectorstore import store_chunks, reset_db
from backend.bm25 import build_bm25
from backend.hybrid import hybrid_search
from backend.llm import generate_answer
from backend.context import build_context


st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 Research Paper RAG System")


# ---------------- UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload Research Paper",
    type=["pdf"]
)

if uploaded_file:

    if st.session_state.get("current_file") != uploaded_file.name:

        st.info("Processing research paper...")

        reset_db()

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        # Extract and chunk
        pages = extract_text_from_pdf(temp_pdf_path)
        chunks = chunk_text(pages)

        # Store vectors
        store_chunks(chunks)

        # Build BM25 index
        build_bm25(chunks)

        # Clean up temporary file
        os.remove(temp_pdf_path)

        st.session_state["indexed"] = True
        st.session_state["current_file"] = uploaded_file.name

        st.success("Research Paper Indexed Successfully!")

        st.subheader("Sample Chunks")

        for i, chunk in enumerate(chunks[:5]):

            st.write(f"### Chunk {i+1}")

            st.write(chunk.page_content)

            st.caption(
                f"Source: {chunk.metadata.get('source','Unknown')} | "
                f"Page: {chunk.metadata.get('page',0)+1} | "
                f"Paragraph: {chunk.metadata.get('paragraph','N/A')}"
            )

            st.divider()


# ---------------- QUERY ----------------

st.subheader("Ask Questions About the Research Paper")

query = st.text_input("Enter your question")

if query:

    if not st.session_state.get("indexed", False):

        st.warning("Please upload a research paper first.")

    else:

        docs = hybrid_search(query, k=5)

        st.subheader("Top Retrieved Context")

        for i, doc in enumerate(docs):

            st.write(f"### Result {i+1}")

            st.write(doc.page_content)

            st.caption(
                f"Source: {doc.metadata.get('source','Unknown')} | "
                f"Page: {doc.metadata.get('page',0)+1} | "
                f"Paragraph: {doc.metadata.get('paragraph','N/A')}"
            )

            st.divider()

        if not docs:

            st.warning("No relevant context found.")

        else:

            context = build_context(docs)

            answer = generate_answer(query, context)

            st.subheader("Answer")

            st.write(answer)