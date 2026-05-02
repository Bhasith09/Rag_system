# streamlit_app 
import streamlit as st
from backend.ingest import extract_text_from_pdf
from backend.chunking import chunk_text
from backend.vectorstore import store_chunks, search, reset_db
from backend.bm25 import build_bm25
from backend.hybrid import hybrid_search
from backend.llm import generate_answer
from backend.context import build_context
from backend.prompts import load_prompt


st.set_page_config(page_title="RAG System", layout="wide")
st.title("🚀 Production RAG System")

# ---------------- RESET DB ----------------
if st.button("Reset Database"):
    reset_db()
    st.success("Database cleared!")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = chunk_text(text)

    st.success(f"Chunks created: {len(chunks)}")

    st.subheader("Sample Chunks")
    for i, chunk in enumerate(chunks[:5]):
        st.write(f"Chunk {i+1}")
        st.write(chunk)
        st.write("---")

    if st.button("Index Document"):
        if "indexed" not in st.session_state:
            store_chunks(chunks)
            build_bm25(chunks)
            st.session_state.indexed=True
            st.success("Document Indexed Successfully!")
        else:
            st.warning("Already indexed this file")

# ---------------- QUERY ----------------
st.subheader("Ask Question")
query = st.text_input("Enter your question")

if query:
    docs = hybrid_search(query, k=5)

    st.write("TOP RETRIEVED CONTEXT:")
    for i, d in enumerate(docs):
        st.write(f"[{i+1}]", d)

    #Build context
    context=build_context(docs)

    #Generate answer
    answer=generate_answer(query,context)

    st.subheader("Answer")
    st.write(answer)