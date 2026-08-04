import hashlib
import streamlit as st

from ingest import extract_text_from_pdf
from chunking import chunk_text
from vectorstore import store_chunks, reset_db
from bm25 import build_bm25
from hybrid import hybrid_search
from context import build_context
from llm import generate_answer

st.set_page_config(page_title="RAG System", layout="wide")
st.title("🚀 Production Hybrid RAG System")

# ---------------- UPLOAD ----------------

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()

    if st.session_state.get("file_hash") != file_hash:

        reset_db()

        # Returns LangChain Documents
        documents = extract_text_from_pdf(uploaded_file)

        # Returns chunked Documents
        chunks = chunk_text(documents)

        st.success(f"Chunks created: {len(chunks)}")

        st.subheader("Sample Chunks")

        for i, chunk in enumerate(chunks[:5], start=1):
            page = chunk.metadata.get("page", 0) + 1
            paragraph = chunk.metadata.get("paragraph", 1)

            st.markdown(f"### Chunk {i}")
            st.write(f"**Page:** {page}")
            st.write(f"**Paragraph:** {paragraph}")
            st.write(chunk.page_content)
            st.write("---")

        # Store in Chroma
        store_chunks(chunks)

        # Build BM25
        build_bm25(chunks)

        st.session_state["file_hash"] = file_hash
        st.session_state["chunks"] = chunks

        st.success("Document Indexed Successfully!")

    else:
        st.info("Document already indexed.")

# ---------------- QUERY ----------------

st.subheader("Ask Question")

query = st.text_input("Enter your question")

if query:

    docs = hybrid_search(query, k=5)

    if not docs:
        st.warning("No relevant context found.")

    else:

        st.subheader("Top Retrieved Context")

        for i, doc in enumerate(docs, start=1):

            page = doc.metadata.get("page", 0) + 1
            paragraph = doc.metadata.get("paragraph", 1)

            st.markdown(f"### [{i}]")
            st.write(f"**Page:** {page}")
            st.write(f"**Paragraph:** {paragraph}")
            st.write(doc.page_content)
            st.write("---")

        context = build_context(docs)

        answer = generate_answer(query, context)

        st.subheader("Answer")
        st.write(answer)