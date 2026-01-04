import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📄",
    layout="wide"
)

# =========================
# Custom CSS
# =========================
st.markdown("""
<style>
.answer-box {
    background-color: #0f172a;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}
.source-box {
    background-color: #020617;
    padding: 12px;
    border-radius: 8px;
    border-left: 4px solid #38bdf8;
    margin-bottom: 10px;
}
small {
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
st.title("📄 RAG Assistant")
st.caption("Upload documents and ask questions using Retrieval-Augmented Generation")

# =========================
# Sidebar — Upload
# =========================
with st.sidebar:
    st.header("📎 Upload documents")

    uploaded_files = st.file_uploader(
        "Upload TXT or PDF files",
        type=["txt", "pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": file}
                )

            if response.status_code == 200:
                st.success(f"{file.name} uploaded")
            else:
                st.error(response.text)

# =========================
# Main — Question
# =========================
st.subheader("💬 Ask a question")

question = st.text_input(
    "Type your question (English or Russian)",
    placeholder="What is RAG?"
)

ask_btn = st.button("Ask", type="primary")

if ask_btn:
    if not question:
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/query",
                json={"question": question}
            )

        if response.status_code != 200:
            st.error(response.text)
        else:
            data = response.json()

            # =========================
            # Answer
            # =========================
            st.subheader("🧠 Answer")
            st.markdown(
                f"<div class='answer-box'>{data['answer']}</div>",
                unsafe_allow_html=True
            )

            # =========================
            # Sources
            # =========================
            if data["sources"]:
                st.subheader("📚 Sources")

                for src in data["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                        <b>{src['source']}</b><br>
                        <small>{src['preview']}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

