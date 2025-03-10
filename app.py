import tempfile
import streamlit as st
from typing import List, Optional

from services.document_service import DocumentService
from services.qa_service import QAService
from config.theme import apply_theme
from config.constants import DEFAULT_TOP_K, SUPPORTED_FILE_TYPES, FILE_TYPE_HELP_TEXT


def main():
    """Main application entry point."""
    apply_theme()
    st.title("Reason about the data!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed" not in st.session_state:
        st.session_state.processed = False

    with st.sidebar:
        st.header("Settings")
        top_k = st.number_input(
            "Retrieval Top K",
            min_value=1,
            max_value=10,
            value=DEFAULT_TOP_K,
            help="Number of document chunks to retrieve for context",
        )
        uploaded_files = handle_multiple_file_upload()

    if uploaded_files and not st.session_state.processed:
        process_uploaded_files(uploaded_files, top_k)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.processed:
        handle_user_input()
    elif not uploaded_files:
        st.info("Please upload one or more documents to start chatting")


def handle_multiple_file_upload():
    """Handle file upload with multiple file support."""
    return st.file_uploader(
        "Upload documents",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help=FILE_TYPE_HELP_TEXT,
    )


def process_uploaded_files(uploaded_files, top_k: int) -> None:
    """Process uploaded files and create a QA chain."""
    document_service = DocumentService()
    vector_store = document_service.process_files(uploaded_files)

    if vector_store:
        retriever = document_service.create_retriever(vector_store, top_k)
        qa_service = QAService()
        qa_chain = qa_service.build_qa_chain(retriever)

        st.session_state.qa_chain = qa_chain
        st.session_state.processed = True
        st.session_state.messages = []
        st.rerun()
    else:
        st.error("No files were successfully processed")
        st.session_state.processed = False


def handle_user_input() -> None:
    """Handle user input and generate responses."""
    if prompt := st.chat_input("Ask about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing..."):
            response = st.session_state.qa_chain.invoke({"input": prompt})["answer"]

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
