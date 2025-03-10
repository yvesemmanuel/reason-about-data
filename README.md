# Reason About Data

A modern document analysis and question answering system that allows you to chat with your data files. This application provides both a user-friendly Streamlit web interface and a scalable FastAPI backend for document processing and question answering.

## Features

- **Multi-format Document Processing**: Upload and analyze PDF, CSV, and TXT files
- **Semantic Document Chunking**: Automatically breaks documents into meaningful chunks for better context retrieval
- **Retrieval-Augmented Question Answering**: Uses LLM to generate accurate answers from context
- **Dual Interface Options**:
  - Interactive Streamlit web app for direct user interaction
  - RESTful API with FastAPI for integration with other applications
- **Asynchronous Background Processing**: Handles document processing in the background for better user experience
- **Session Management**: Manages document processing results with session-based access

## Architecture

The system follows a modular design with these key components:

1. **Document Processing**: 
   - Processes various document formats (PDF, CSV, TXT)
   - Uses semantic chunking to divide documents into meaningful segments
   - Creates vector embeddings for efficient retrieval

2. **Vector Storage**:
   - Utilizes FAISS for efficient similarity search
   - Combines multiple document vector stores for unified querying

3. **Question Answering Chain**:
   - Retrieves relevant document chunks based on query
   - Formats document chunks with source information
   - Uses Retrieval-Augmented Generation (RAG) for accurate answers

4. **User Interfaces**:
   - Streamlit web app for interactive usage
   - FastAPI backend for programmatic access

## Installation

```bash
# Clone the repository
git clone https://github.com/yvesemmanuel/reason-about-data.git
cd reason-about-data

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Streamlit App

```bash
streamlit run app.py
```

This will launch the web interface at http://localhost:8501 where you can:
1. Upload documents (PDF, CSV, TXT)
2. Ask questions about the uploaded documents
3. View answers generated based on document context

### Starting the FastAPI Server

```bash
uvicorn api:app --reload
```

This will start the API server at http://localhost:8000 with automatic reload on code changes.

## API Endpoints

### API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Upload Documents
```
POST /documents/upload
```
- Upload one or more documents for processing
- Returns a session ID for subsequent queries

#### Query Documents
```
POST /query
```
- Ask questions about uploaded documents
- Requires the session ID from document upload

#### Check Processing Status
```
GET /session/{session_id}/status
```
- Check if document processing is complete

## Example API Usage

### 1. Upload Documents

```python
import requests

# Upload documents
files = [
    ('files', ('document.pdf', open('document.pdf', 'rb'), 'application/pdf')),
    ('files', ('data.csv', open('data.csv', 'rb'), 'text/csv'))
]

response = requests.post('http://localhost:8000/documents/upload', files=files)
session_id = response.json()['session_id']
print(f"Session ID: {session_id}")
```

### 2. Query Documents

```python
# Wait for processing to complete
import time
import requests

status = "processing"
while status == "processing":
    status_response = requests.get(f'http://localhost:8000/session/{session_id}/status')
    status = status_response.json()['status']
    if status == "processing":
        print("Still processing...")
        time.sleep(2)

# Query the documents
query_data = {
    "session_id": session_id,
    "query": "What are the main findings in these documents?"
}

query_response = requests.post('http://localhost:8000/query', json=query_data)
print(query_response.json()['answer'])
```

## Technical Details

### Model Configuration

The application uses Ollama to run the language models locally. The default model is configured in `config/model.py`:

```python
OLLAMA_MODEL_ID = "deepseek-r1:8b"
```

You can change this to any other model supported by Ollama.

### Document Processing

Documents are processed using LangChain's document loaders and text splitters:

1. **Loading**: PDFPlumberLoader, CSVLoader, TextLoader
2. **Embedding**: HuggingFaceEmbeddings for vector creation
3. **Chunking**: SemanticChunker for context-aware document splitting
4. **Indexing**: FAISS for fast similarity search

### Question Answering

The QA pipeline follows these steps:

1. **Query Reception**: User submits a question
2. **Context Retrieval**: System retrieves relevant document chunks
3. **Context Formatting**: Retrieved chunks are formatted with metadata
4. **Answer Generation**: LLM generates an answer based on the retrieved context
5. **Response**: Answer is returned to the user

## Project Structure

```
reason-about-data/
├── app.py               # Streamlit application
├── api.py               # FastAPI application
├── config/              # Configuration files
│   ├── colors.py        # UI color theme
│   ├── constants.py     # Application constants
│   ├── model.py         # Model configuration
│   └── theme.py         # Streamlit theme
├── services/            # Business logic services
│   ├── document_service.py  # Document processing
│   └── qa_service.py        # Question answering
├── requirements.txt     # Dependencies
└── README.md            # Documentation
```

## Development

### Testing

```bash
# Run tests
pytest
```

### Code Quality

```bash
# Run linter
ruff check .
```

## Requirements

- Python 3.9+
- Ollama (for local LLM)
- Dependencies listed in requirements.txt

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request 