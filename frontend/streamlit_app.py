import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 RAG Assistant")
st.caption("Upload documents and ask questions using RAG")

# --- Upload section ---
st.header("1️⃣ Upload document")

uploaded_file = st.file_uploader(
    "Upload TXT or PDF",
    type=["txt", "pdf"]
)

if uploaded_file:
    with st.spinner("Uploading and processing document..."):
        response = requests.post(
            f"{API_URL}/upload",
            files={"file": uploaded_file}
        )

    if response.status_code == 200:
        st.success("Document processed successfully")
    else:
        st.error(response.text)

# --- Query section ---
st.header("2️⃣ Ask a question")

question = st.text_input("Your question")

if st.button("Ask"):
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question}
            )

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Sources")
            for src in data["sources"]:
                st.markdown(
                    f"- **{src['source']}**: {src['preview']}"
                )
        else:
            st.error(response.text)
