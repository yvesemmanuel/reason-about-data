import tempfile
import streamlit as st
from config.colors import PRIMARY, SECONDARY, BACKGROUND, TEXT
from processing.document_processor import DocumentProcessor
from processing.qa_chain import QAChainBuilder


def apply_theme():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {BACKGROUND};
            color: {TEXT};
        }}
        .stChatInput {{
            position: fixed;
            bottom: 2rem;
            width: 80%;
        }}
        .stChatMessage {{
            border-radius: 15px;
            padding: 10px 20px;
            margin: 5px 0;
            max-width: 80%;
        }}
        .stChatMessage.user {{
            background-color: {PRIMARY};
            color: white;
            margin-left: auto;
        }}
        .stChatMessage.assistant {{
            background-color: {SECONDARY};
            color: white;
            margin-right: auto;
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )


def main():
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
            value=3,
            help="Number of document chunks to retrieve for context",
        )
        uploaded_files = handle_multiple_file_upload()

    if uploaded_files and not st.session_state.processed:
        processor = DocumentProcessor()
        combined_vector_store = None

        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_extension}"
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                current_vector_store = process_single_file(
                    processor, tmp_path, file_extension
                )

                if combined_vector_store is None:
                    combined_vector_store = current_vector_store
                else:
                    combined_vector_store.merge_from(current_vector_store)

            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue

        if combined_vector_store is not None:
            retriever = processor.create_retriever(combined_vector_store, top_k)
            qa_chain = QAChainBuilder.build_chain(retriever)

            st.session_state.qa_chain = qa_chain
            st.session_state.processed = True
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("No files were successfully processed")
            st.session_state.processed = False

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.processed:
        if prompt := st.chat_input("Ask about the documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Analyzing..."):
                response = st.session_state.qa_chain.invoke({"input": prompt})["answer"]

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
    elif not uploaded_files:
        st.info(
            "Please upload one or more documents (PDF, CSV, or TXT) to start chatting"
        )


def handle_multiple_file_upload():
    return st.file_uploader(
        "Upload documents",
        type=["pdf", "csv", "txt"],
        accept_multiple_files=True,
        help="Upload multiple PDF, CSV, or text files for analysis",
    )


def process_single_file(
    processor: DocumentProcessor, file_path: str, file_extension: str
):
    if file_extension == "pdf":
        return processor.process_pdf(file_path)
    elif file_extension == "csv":
        return processor.process_csv(file_path)
    elif file_extension == "txt":
        return processor.process_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


if __name__ == "__main__":
    main()
