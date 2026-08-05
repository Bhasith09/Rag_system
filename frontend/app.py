import streamlit as st
import requests
API_URL = "API_URL"  # Use the environment variable for the API URL
#API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Production Hybrid RAG", layout="wide")
st.title("🚀 Production Hybrid RAG System")

# ---------------- UPLOAD ----------------

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:

    try:
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                "application/pdf"
            )
        }

        response = requests.post(
            f"{API_URL}/upload",
            files=files
        )

        if response.status_code == 200:
            data = response.json()

            st.success(data["message"])
            st.write(f"Chunks Created: {data['chunks']}")

        else:
            st.error(f"Upload Failed (Status Code: {response.status_code})")

            try:
                st.json(response.json())
            except Exception:
                st.code(response.text)

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI server.")
        st.info("Start the backend using:\n\nuvicorn backend.main:app --reload")

# ---------------- QUESTION ----------------

st.subheader("Ask Question")

query = st.text_input("Enter your question")

if st.button("Ask"):

    if not query.strip():
        st.warning("Please enter a question.")

    else:

        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "question": query
                }
            )

            if response.status_code == 200:

                data = response.json()

                st.subheader("Top Retrieved Context")

                for i, doc in enumerate(data["contexts"], start=1):

                    st.markdown(f"### [{i}]")
                    st.write(f"**Page:** {doc['page']}")
                    st.write(f"**Paragraph:** {doc['paragraph']}")
                    st.write(doc["text"])
                    st.write("---")

                st.subheader("Answer")
                st.write(data["answer"])

            else:

                st.error(f"Request Failed (Status Code: {response.status_code})")

                try:
                    st.json(response.json())
                except Exception:
                    st.code(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI server.")
            st.info("Start the backend using:\n\nuvicorn backend.main:app --reload")