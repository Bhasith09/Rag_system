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
st.title("📚 Research Paper RAG System")


# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Research Paper", type=["pdf"])

if uploaded_file:

    if st.session_state.get("current_file") != uploaded_file.name:
        st.info("Processing research paper...")
        reset_db()  #reset database when new file is uploaded

        
        pages = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(pages)


        # Step 1: store embeddings
        store_chunks(chunks)

        # Step 3: build BM25 index
        build_bm25(chunks)
#save session state
        st.session_state["indexed"] = True
        st.session_state["current_file"] = uploaded_file.name
        
        st.success("Research Paper Indexed Successfully!")
        st.subheader("Sample Chunks")
        for i, chunk in enumerate(chunks[:5]):
            st.write(f"Chunk {i+1}")

            st.write(chunk["text"])

            st.write(
                f"Page: {chunk['page']} | Paragraph: {chunk['paragraph']}"
            )

            st.write("---")


# ---------------- QUERY ----------------
st.subheader("Ask Questions About the Research Paper")
query = st.text_input("Enter your question")

if query:
    if not st.session_state.get("indexed", False):
        st.warning("Please upload a research paper first.")
    else:
        docs = hybrid_search(query, k=5)

        st.write("TOP RETRIEVED CONTEXT:")
        for i, d in enumerate(docs):
            st.write(f"[{i+1}]")
            st.write(d["text"])
            st.write(
                f"Source: {d['source']} | "
                f"Page: {d['page']} | "
                f"Paragraph: {d['paragraph']}"
            )
            st.write("---")
        #generate answer
        if not docs:
            st.write("No relevant context found")
        else:

            context=build_context(docs)
            answer=generate_answer(query,context)

            st.subheader("Answer")
            st.write(answer)