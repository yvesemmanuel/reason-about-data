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
            overflow-x: hidden;
        }}
        .stChatInput {{
            position: fixed;
            bottom: 2rem;
            width: 80%;
            max-width: 900px;
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
        /* Fix horizontal scrollbar */
        .main .block-container {{
            max-width: 1000px;
            padding-top: 2rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 3rem;
            overflow-x: hidden;
        }}
        /* Improve sidebar appearance */
        .sidebar .sidebar-content {{
            width: 100%;
            overflow-y: auto;
            overflow-x: hidden;
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )
