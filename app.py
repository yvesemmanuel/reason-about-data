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


def handle_file_upload():
    uploaded_file = st.file_uploader("Upload a PDF file!", type="pdf")
    if not uploaded_file:
        st.info("Please upload a PDF file to proceed.")
        return None
    return uploaded_file


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
        uploaded_file = handle_file_upload()

    if uploaded_file and not st.session_state.processed:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            processor = DocumentProcessor()
            vector_store = processor.process_pdf(tmp.name)
            retriever = processor.create_retriever(vector_store, top_k)
            qa_chain = QAChainBuilder.build_chain(retriever)

            st.session_state.qa_chain = qa_chain
            st.session_state.processed = True
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.processed:
        if prompt := st.chat_input("Ask about the PDF..."):
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Analyzing..."):
                response = st.session_state.qa_chain.invoke({"input": prompt})["answer"]

            st.session_state.messages.append({"role": "assistant", "content": response})

            st.rerun()
    elif not uploaded_file:
        st.info("Please upload a PDF file to start chatting")


if __name__ == "__main__":
    main()
