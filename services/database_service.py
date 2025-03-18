"""Database service for conversation storage.

This module handles the SQLite database operations for storing and retrieving
user conversations with UUID tracking.
"""

import sqlite3
import uuid
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from config.settings import get_db_config


class DatabaseService:
    """Service for managing conversation data in SQLite database."""

    def __init__(self):
        """Initialize the database service.

        Creates the database and tables if they don't exist.
        """
        db_config = get_db_config()
        self.db_path = db_config["db_path"]

        if db_config["create_tables"]:
            self._create_tables()

    def _create_tables(self) -> None:
        """Create the necessary database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    title TEXT,
                    model_id TEXT,
                    mode TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)

            conn.commit()

    def create_conversation(
        self, title: str, model_id: str, mode: str
    ) -> Tuple[Optional[str], Optional[Exception]]:
        """Create a new conversation.

        Parameters:
        -----------
        title: str
            Title of the conversation
        model_id: str
            ID of the model used
        mode: str
            Mode of the conversation (direct or rag)

        Returns:
        --------
        Tuple[Optional[str], Optional[Exception]]: The UUID of the created conversation and error
        """
        try:
            conversation_id = str(uuid.uuid4())
            current_time = datetime.now().isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO conversations (
                        conversation_id, created_at, updated_at, title, model_id, mode
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        current_time,
                        current_time,
                        title,
                        model_id,
                        mode,
                    ),
                )
                conn.commit()

            return conversation_id, None
        except Exception as e:
            return None, e

    def get_conversation(
        self, conversation_id: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
        """Get a conversation by ID.

        Parameters:
        -----------
        conversation_id: str
            UUID of the conversation

        Returns:
        --------
        Tuple[Optional[Dict[str, Any]], Optional[Exception]]: Conversation data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT * FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                )
                row = cursor.fetchone()

                if not row:
                    return None, None

                return dict(row), None
        except Exception as e:
            return None, e

    def list_conversations(self) -> Tuple[List[Dict[str, Any]], Optional[Exception]]:
        """List all conversations.

        Returns:
        --------
        Tuple[List[Dict[str, Any]], Optional[Exception]]: List of all conversations and error
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
                rows = cursor.fetchall()

                return [dict(row) for row in rows], None
        except Exception as e:
            return [], e

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[Exception]]:
        """Add a message to a conversation.

        Parameters:
        -----------
        conversation_id: str
            UUID of the conversation
        role: str
            Role of the message sender (user or assistant)
        content: str
            Content of the message
        metadata: Optional[Dict[str, Any]]
            Optional metadata for the message

        Returns:
        --------
        Tuple[Optional[str], Optional[Exception]]: The UUID of the created message and error
        """
        message_id = str(uuid.uuid4())
        current_time = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else "{}"

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    INSERT INTO messages (
                        message_id, conversation_id, role, content, created_at, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        conversation_id,
                        role,
                        content,
                        current_time,
                        metadata_json,
                    ),
                )

                cursor.execute(
                    "UPDATE conversations SET updated_at = ? WHERE conversation_id = ?",
                    (current_time, conversation_id),
                )

                conn.commit()

            return message_id, None
        except Exception as e:
            return None, e

    def get_messages(
        self, conversation_id: str
    ) -> Tuple[List[Dict[str, Any]], Optional[Exception]]:
        """Get all messages in a conversation.

        Parameters:
        -----------
        conversation_id: str
            UUID of the conversation

        Returns:
        --------
        Tuple[List[Dict[str, Any]], Optional[Exception]]: List of messages in the conversation and error
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at",
                    (conversation_id,),
                )
                rows = cursor.fetchall()

                messages = []
                for row in rows:
                    row_dict = dict(row)
                    try:
                        row_dict["metadata"] = json.loads(row_dict["metadata"])
                    except (json.JSONDecodeError, TypeError):
                        row_dict["metadata"] = {}
                    messages.append(row_dict)

                return messages, None
        except Exception as e:
            return [], e

    def delete_conversation(
        self, conversation_id: str
    ) -> Tuple[bool, Optional[Exception]]:
        """Delete a conversation and its messages.

        Parameters:
        -----------
        conversation_id: str
            UUID of the conversation

        Returns:
        --------
        Tuple[bool, Optional[Exception]]: True if the conversation was deleted, False otherwise and error
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
                )

                cursor.execute(
                    "DELETE FROM conversations WHERE conversation_id = ?",
                    (conversation_id,),
                )

                conn.commit()

                return cursor.rowcount > 0, None
        except Exception as e:
            return False, e
