"""Main application entry point.

This module initializes and runs the Streamlit application with proper
configuration based on the environment settings.
"""

import streamlit as st

from services.document_service import DocumentService
from services.qa_service import QAService
from services.database_service import DatabaseService
from config.theme import apply_theme
from config.settings import ENV, get_app_config
from config.constants import (
    SUPPORTED_FILE_TYPES,
    FILE_TYPE_HELP_TEXT,
    DIRECT_CHAT_INFO,
    RAG_CHAT_SUCCESS,
    DIRECT_CHAT_PROMPT,
    DOCUMENT_CHAT_PROMPT,
)
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


def initialize_app() -> DatabaseService:
    """Initialize application components and return database service."""
    apply_theme()
    get_app_config()
    return DatabaseService()


def initialize_base_state(
    initialized: bool = True,
    messages: list = [],
    processed: bool = False,
    initial_mode: str = "direct",
    initial_conversation_id: str = None,
):
    """Initialize core session state variables.

    Parameters:
    -----------
    initialized: bool, default=True
        Whether the session state is initialized
    messages: list, default=[]
        The list of messages
    processed: bool, default=False
        Whether the conversation has been processed
    initial_mode: str, default="direct"
        The initial mode of the conversation
    initial_conversation_id: str, default=None
        The initial conversation ID
    """
    st.session_state.setdefault("initialized", initialized)
    st.session_state.setdefault("messages", messages)
    st.session_state.setdefault("processed", processed)
    st.session_state.setdefault("mode", initial_mode)
    st.session_state.setdefault("conversation_id", initial_conversation_id)
    st.session_state.setdefault("mode_ready", initial_mode == "direct")


def initialize_model_settings(
    selected_model: str,
    temperature: float,
    top_k: int,
    max_tokens: int,
    top_p: float,
):
    """Initialize model-related session state variables.

    Parameters:
    -----------
    selected_model: str
        The selected model
    temperature: float
        The temperature
    top_k: int
        The top-k value
    max_tokens: int
        The maximum number of tokens
    top_p: float
        The top-p value
    """
    st.session_state.setdefault("selected_model", selected_model)
    st.session_state.setdefault("temperature", temperature)
    st.session_state.setdefault("top_k", top_k)
    st.session_state.setdefault("max_tokens", max_tokens)
    st.session_state.setdefault("top_p", top_p)


def create_direct_chain() -> None:
    """Create and store the direct conversation chain in session state."""
    qa_service = QAService()
    direct_chain, error = qa_service.build_direct_chain(
        st.session_state.selected_model,
        st.session_state.temperature,
        st.session_state.max_tokens,
        st.session_state.top_p,
    )
    if error:
        st.error(f"Error building direct chain: {error}")
        return

    st.session_state.direct_chain = direct_chain


def setup_main_layout(db_service: DatabaseService) -> None:
    """Configure and display the main application layout."""
    container = st.container()
    with container:
        display_conversation_history(db_service)
        show_mode_specific_alerts()

        if st.session_state.mode == "rag" and not st.session_state.processed:
            st.subheader("Document Upload")
            st.write("Please upload documents to start the Document Chat")
            uploaded_files = st.file_uploader(
                "Upload documents",
                type=SUPPORTED_FILE_TYPES,
                accept_multiple_files=True,
                help=FILE_TYPE_HELP_TEXT,
                key="main_file_uploader",
            )

            if uploaded_files:
                process_button = st.button("Process Documents", type="primary")
                if process_button:
                    with st.spinner("Processing documents..."):
                        process_uploaded_files(uploaded_files)

        handle_sidebar_operations()

        if (st.session_state.mode == "direct") or (
            st.session_state.mode == "rag" and st.session_state.processed
        ):
            handle_chat_input(db_service)


def show_mode_specific_alerts() -> None:
    """Display alerts based on current conversation mode and state."""
    if st.session_state.mode == "direct":
        st.info(DIRECT_CHAT_INFO)
    elif st.session_state.mode == "rag":
        if st.session_state.processed:
            st.success(RAG_CHAT_SUCCESS)
        else:
            st.warning(
                "Document Chat requires uploading and processing documents before chatting. Please upload documents above."
            )


def reset_conversation_state():
    """Reset the conversation state to default values."""
    st.session_state.messages = []
    st.session_state.processed = False
    st.session_state.conversation_id = None


def handle_mode_selection() -> None:
    """Handle conversation mode selection and state reset."""
    mode = st.radio(
        "Mode",
        options=["Direct Chat", "Document Chat"],
        index=0 if st.session_state.mode == "direct" else 1,
        help="Choose between direct conversation with the model or retrieval-based chat with documents.",
    )

    new_mode = "direct" if mode == "Direct Chat" else "rag"
    if new_mode != st.session_state.mode:
        previous_mode = st.session_state.mode

        st.session_state.mode = new_mode

        reset_conversation_state()

        initialize_mode_environment(new_mode, previous_mode)

        st.rerun()


def initialize_mode_environment(new_mode: str, previous_mode: str) -> None:
    """Proactively prepare the environment for the selected mode.

    Parameters:
    -----------
    new_mode: str
        The new conversation mode ('direct' or 'rag')
    previous_mode: str
        The previous conversation mode
    """
    if previous_mode == "rag" and "qa_chain" in st.session_state:
        del st.session_state.qa_chain
        st.session_state.processed = False

    if new_mode == "direct":
        if "direct_chain" not in st.session_state:
            create_direct_chain()
        st.session_state.mode_ready = True
    else:
        st.session_state.processed = False
        st.session_state.mode_ready = False


def create_model_settings() -> None:
    """Create model configuration controls in the sidebar."""
    selected_model = st.selectbox(
        "LLM Model",
        options=AVAILABLE_MODELS,
        index=DEFAULT_MODEL_INDEX,
        help="Choose which language model to use for answering questions",
    )

    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        create_direct_chain()


def update_model_chain(param_name: str, value: float) -> None:
    """Update model chain when parameter changes."""
    if getattr(st.session_state, param_name) != value:
        setattr(st.session_state, param_name, value)
        create_direct_chain()


def handle_retrieval_settings() -> None:
    """Handle retrieval-specific settings for RAG mode."""
    if st.session_state.mode == "rag":
        st.subheader("Retrieval Settings")
        st.session_state.top_k = st.slider(
            "Retrieval Context Size",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
            step=1,
            help="Number of document chunks to retrieve for context.",
        )


def handle_sidebar_operations() -> None:
    """Handle all sidebar operations and file uploads."""
    with st.sidebar:
        st.header("Settings")
        handle_mode_selection()

        st.subheader("Model Settings")
        create_model_settings()

        update_model_chain(
            "temperature",
            st.slider(
                "Temperature",
                MIN_TEMPERATURE,
                MAX_TEMPERATURE,
                DEFAULT_TEMPERATURE,
                TEMPERATURE_STEP,
                help="Controls randomness in response generation.",
            ),
        )

        update_model_chain(
            "top_p",
            st.slider(
                "Top-p (Nucleus Sampling)",
                MIN_TOP_P,
                MAX_TOP_P,
                DEFAULT_TOP_P,
                TOP_P_STEP,
                help="Controls diversity through token probability.",
            ),
        )

        update_model_chain(
            "max_tokens",
            st.slider(
                "Max Output Tokens",
                MIN_MAX_TOKENS,
                MAX_MAX_TOKENS,
                DEFAULT_MAX_TOKENS,
                MAX_TOKENS_STEP,
                help="Maximum number of tokens to generate.",
            ),
        )

        handle_retrieval_settings()


def display_messages(messages: list) -> None:
    """Display conversation messages in chat format."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def display_conversation_history(db_service: DatabaseService) -> None:
    """Display conversation history from database or session state."""
    if st.session_state.conversation_id:
        messages, error = db_service.get_messages(st.session_state.conversation_id)
        if error:
            st.error(f"Error fetching messages: {error}")
            return
    else:
        messages = st.session_state.messages

    display_messages(messages)


def process_uploaded_files(uploaded_files) -> None:
    """Process uploaded files and initialize QA chain."""
    if not st.session_state.processed:
        with st.status("Processing documents...", expanded=True) as status:
            st.write("Extracting content from documents...")
            vector_store, error = DocumentService().process_files(uploaded_files)

            if error:
                st.error(f"Error processing files: {error}")
                status.update(label="Processing failed", state="error")
                return

            if not vector_store:
                st.error(
                    "No files were successfully processed. Please check your documents and try again."
                )
                status.update(label="Processing failed", state="error")
                return

            st.write("Creating retriever...")
            retriever, error = DocumentService().create_retriever(
                vector_store, st.session_state.top_k
            )
            if error:
                st.error(f"Error creating retriever: {error}")
                status.update(label="Processing failed", state="error")
                return

            st.write("Building QA chain...")
            qa_chain, error = QAService().build_qa_chain(
                retriever,
                st.session_state.selected_model,
                st.session_state.temperature,
                st.session_state.max_tokens,
                st.session_state.top_p,
            )
            if error:
                st.error(f"Error building QA chain: {error}")
                status.update(label="Processing failed", state="error")
                return

            st.session_state.update(
                {
                    "qa_chain": qa_chain,
                    "processed": True,
                    "mode_ready": True,
                    "messages": [],
                    "conversation_id": None,
                }
            )

            status.update(label="Documents processed successfully!", state="complete")

        st.success("Your documents are ready! You can now start chatting with them.")
        st.rerun()


def create_conversation_if_needed(db_service: DatabaseService, prompt: str) -> None:
    """Create new conversation record if none exists."""
    if not st.session_state.conversation_id:
        title = prompt[:30] + "..." if len(prompt) > 30 else prompt
        st.session_state.conversation_id, error = db_service.create_conversation(
            title=title,
            model_id=st.session_state.selected_model,
            mode=st.session_state.mode,
        )
        if error:
            st.error(f"Error creating conversation: {error}")
            return


def generate_response(prompt: str) -> str:
    """Generate response based on current mode and prompt.

    Parameters:
    -----------
    prompt: str
        The user's prompt

    Returns:
    --------
    str: The generated response
    """
    if st.session_state.mode == "direct":
        response, error = QAService().query_direct(
            st.session_state.direct_chain, prompt
        )
        if error:
            st.error(f"Error querying direct chain: {error}")
            return ""
        return response["answer"]

    return st.session_state.qa_chain.invoke({"input": prompt})["answer"]


def save_response(db_service: DatabaseService, prompt: str, response: str) -> None:
    """Save conversation messages to database and session state."""
    _, error = db_service.add_message(st.session_state.conversation_id, "user", prompt)
    if error:
        st.error(f"Error adding user message: {error}")
        return

    _, error = db_service.add_message(
        st.session_state.conversation_id, "assistant", response
    )
    if error:
        st.error(f"Error adding assistant message: {error}")
        return

    st.session_state.messages.extend(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    )


def handle_chat_input(db_service: DatabaseService) -> None:
    """Handle user chat input and generate responses."""
    prompt_text = (
        DIRECT_CHAT_PROMPT
        if st.session_state.mode == "direct"
        else DOCUMENT_CHAT_PROMPT
    )

    if prompt := st.chat_input(prompt_text):
        create_conversation_if_needed(db_service, prompt)

        with st.spinner("Thinking..."):
            response = generate_response(prompt)

        save_response(db_service, prompt, response)
        st.rerun()


def main():
    """Main application entry point."""
    db_service = initialize_app()
    initialize_base_state(initial_mode="direct")
    initialize_model_settings(
        selected_model=AVAILABLE_MODELS[DEFAULT_MODEL_INDEX],
        temperature=DEFAULT_TEMPERATURE,
        top_k=DEFAULT_TOP_K,
        max_tokens=DEFAULT_MAX_TOKENS,
        top_p=DEFAULT_TOP_P,
    )
    create_direct_chain()

    st.title(f"Reason about the data!{f' ({ENV})' if ENV != 'production' else ''}")
    setup_main_layout(db_service)


if __name__ == "__main__":
    main()
