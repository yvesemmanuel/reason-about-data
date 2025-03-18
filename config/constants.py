"""Application constants and configuration values."""

SUPPORTED_FILE_TYPES = ["pdf", "csv", "txt"]
FILE_TYPE_HELP_TEXT = "Upload multiple PDF, CSV, or text files for analysis"

WELCOME_MESSAGE = "Please upload one or more documents to start chatting"
PROCESSING_MESSAGE = "Analyzing..."
ERROR_NO_FILES_PROCESSED = "No files were successfully processed"

# Chat mode descriptions
DIRECT_CHAT_INFO = "📝 Direct Chat Mode: Chat with the model without document context"
RAG_CHAT_SUCCESS = "📚 Document Chat Mode: Answers based on your uploaded documents"

# Input prompts
DIRECT_CHAT_PROMPT = "Ask a question..."
DOCUMENT_CHAT_PROMPT = "Ask about the documents..."
