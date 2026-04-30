# streamlit_app 
import streamlit as st
from backend.ingest import extract_text_from_pdf
from backend.chunking import chunk_text
from backend.vectorstore import store_chunks, search, reset_db

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
        store_chunks(chunks)
        st.success("Document Indexed Successfully!")

# ---------------- QUERY ----------------
st.subheader("Ask Question")
query = st.text_input("Enter your question")

if query:
    results = search(query)

    st.subheader("Top Retrieved Chunks")

    if not results:
        st.warning("No results found")
    else:
        for i, r in enumerate(results):
            st.write(f"Result {i+1}")
            st.write(r)
            st.write("---")