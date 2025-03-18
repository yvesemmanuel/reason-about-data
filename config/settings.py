"""Application configuration settings.

This module loads configuration from environment variables and provides
configuration settings for different environments (development, production, testing).
"""

import os
from typing import Dict, Any
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent
ENV = os.getenv("ENV", "development")
DATA_DIR = BASE_DIR / "data"
os.makedirs(DATA_DIR, exist_ok=True)

DB_CONFIG = {
    "development": {
        "db_path": str(DATA_DIR / "conversations_dev.db"),
        "create_tables": True,
    },
    "testing": {
        "db_path": ":memory:",
        "create_tables": True,
    },
    "production": {
        "db_path": str(DATA_DIR / "conversations.db"),
        "create_tables": True,
    },
}

STREAMLIT_CONFIG = {
    "development": {
        "server_port": int(os.getenv("STREAMLIT_SERVER_PORT", 8501)),
        "server_address": os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost"),
        "debug": True,
    },
    "testing": {
        "server_port": int(os.getenv("STREAMLIT_SERVER_PORT", 8501)),
        "server_address": os.getenv("STREAMLIT_SERVER_ADDRESS", "localhost"),
        "debug": True,
    },
    "production": {
        "server_port": int(os.getenv("STREAMLIT_SERVER_PORT", 8501)),
        "server_address": os.getenv("STREAMLIT_SERVER_ADDRESS", "0.0.0.0"),
        "debug": False,
    },
}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-r1:1.5b")

APP_CONFIG = {
    "development": {
        "enable_logging": True,
        "log_level": "DEBUG",
    },
    "testing": {
        "enable_logging": True,
        "log_level": "DEBUG",
    },
    "production": {
        "enable_logging": True,
        "log_level": "INFO",
    },
}


def get_db_config() -> Dict[str, Any]:
    """Get database configuration for the current environment.

    Returns:
    --------
        Dict[str, Any]: Database configuration
    """
    return DB_CONFIG.get(ENV, DB_CONFIG["development"])


def get_streamlit_config() -> Dict[str, Any]:
    """Get Streamlit configuration for the current environment.

    Returns:
    --------
        Dict[str, Any]: Streamlit configuration
    """
    return STREAMLIT_CONFIG.get(ENV, STREAMLIT_CONFIG["development"])


def get_app_config() -> Dict[str, Any]:
    """Get application configuration for the current environment.

    Returns:
    --------
        Dict[str, Any]: Application configuration
    """
    return APP_CONFIG.get(ENV, APP_CONFIG["development"])
