import streamlit as st


from services.document_service import DocumentService
from services.qa_service import QAService
from config.theme import apply_theme
from config.constants import SUPPORTED_FILE_TYPES, FILE_TYPE_HELP_TEXT
from config.model import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL_INDEX,
    DEFAULT_TEMPERATURE,
    MIN_TEMPERATURE,
    MAX_TEMPERATURE,
    TEMPERATURE_STEP,
    MIN_TOP_K,
    MAX_TOP_K,
    DEFAULT_TOP_K,
    DEFAULT_MAX_TOKENS,
    MIN_MAX_TOKENS,
    MAX_MAX_TOKENS,
    MAX_TOKENS_STEP,
    DEFAULT_TOP_P,
    MIN_TOP_P,
    MAX_TOP_P,
    TOP_P_STEP,
)


def main():
    """Main application entry point."""
    apply_theme()
    st.title("Reason about the data!")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processed" not in st.session_state:
        st.session_state.processed = False
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[DEFAULT_MODEL_INDEX]
    if "temperature" not in st.session_state:
        st.session_state.temperature = DEFAULT_TEMPERATURE
    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = DEFAULT_MAX_TOKENS
    if "top_p" not in st.session_state:
        st.session_state.top_p = DEFAULT_TOP_P

    container = st.container()

    sidebar = st.sidebar
    with sidebar:
        st.header("Settings")

        st.subheader("Model Settings")

        selected_model = st.selectbox(
            "LLM Model",
            options=AVAILABLE_MODELS,
            index=DEFAULT_MODEL_INDEX,
            help="Choose which language model to use for answering questions",
        )
        st.session_state.selected_model = selected_model

        temperature = st.slider(
            "Temperature",
            min_value=MIN_TEMPERATURE,
            max_value=MAX_TEMPERATURE,
            value=DEFAULT_TEMPERATURE,
            step=TEMPERATURE_STEP,
            help="Controls randomness in response generation. Lower values are more deterministic, higher values more creative.",
        )
        st.session_state.temperature = temperature

        top_p = st.slider(
            "Top-p (Nucleus Sampling)",
            min_value=MIN_TOP_P,
            max_value=MAX_TOP_P,
            value=DEFAULT_TOP_P,
            step=TOP_P_STEP,
            help="Controls diversity by only considering tokens with cumulative probability < top_p. Lower values focus on higher probability tokens.",
        )
        st.session_state.top_p = top_p

        max_tokens = st.slider(
            "Max Output Tokens",
            min_value=MIN_MAX_TOKENS,
            max_value=MAX_MAX_TOKENS,
            value=DEFAULT_MAX_TOKENS,
            step=MAX_TOKENS_STEP,
            help="Maximum number of tokens to generate for the response. Higher values allow longer responses.",
        )
        st.session_state.max_tokens = max_tokens

        st.subheader("Retrieval Settings")

        top_k = st.slider(
            "Retrieval Context Size",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
            step=1,
            help="Number of document chunks to retrieve for context. Higher values provide more context but may increase noise.",
        )
        st.session_state.top_k = top_k

        st.subheader("Document Upload")
        uploaded_files = handle_multiple_file_upload()

    # Use the container for main content to fix horizontal scrolling
    with container:
        if uploaded_files and not st.session_state.processed:
            process_uploaded_files(
                uploaded_files,
                st.session_state.top_k,
                st.session_state.selected_model,
                st.session_state.temperature,
                st.session_state.max_tokens,
                st.session_state.top_p,
            )

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.session_state.processed:
            handle_user_input()
        elif not uploaded_files:
            st.info("Please upload one or more documents to start chatting")


def handle_multiple_file_upload():
    """Handle file upload with multiple file support.

    Returns:
    --------
        List of uploaded files from Streamlit
    """
    return st.file_uploader(
        "Upload documents",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        help=FILE_TYPE_HELP_TEXT,
    )


def process_uploaded_files(
    uploaded_files,
    top_k: int,
    model_id: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> None:
    """Process uploaded files and create a QA chain.

    Parameters:
    -----------
        uploaded_files: List of uploaded files from Streamlit
        top_k: Number of document chunks to retrieve
        model_id: Language model ID to use
        temperature: Controls randomness in response generation (0.0-1.0)
        max_tokens: Maximum number of tokens to generate
        top_p: Top-p sampling parameter (nucleus sampling)

    Returns:
    --------
        None
    """
    document_service = DocumentService()
    vector_store = document_service.process_files(uploaded_files)

    if vector_store:
        retriever = document_service.create_retriever(vector_store, top_k)
        qa_service = QAService()
        qa_chain = qa_service.build_qa_chain(
            retriever,
            model_id,
            temperature,
            max_tokens,
            top_p,
        )

        st.session_state.qa_chain = qa_chain
        st.session_state.processed = True
        st.session_state.messages = []
        st.rerun()
    else:
        st.error("No files were successfully processed")
        st.session_state.processed = False


def handle_user_input() -> None:
    """Handle user input and generate responses.

    Parameters:
    -----------
        None

    Returns:
    --------
        None
    """
    if prompt := st.chat_input("Ask about the documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Analyzing..."):
            response = st.session_state.qa_chain.invoke({"input": prompt})["answer"]

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()


if __name__ == "__main__":
    main()
