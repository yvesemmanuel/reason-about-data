"""UI theme configuration."""

import streamlit as st
from .colors import PRIMARY, SECONDARY, BACKGROUND, TEXT


def apply_theme():
    """Apply custom theme to Streamlit UI."""
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
