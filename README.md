# Reason About Data

A powerful Retrieval-Augmented Generation (RAG) system that allows users to upload documents and interact with them through a natural language interface. The application leverages Ollama language models and FAISS vector database for efficient document retrieval and question answering.

## Features

- **Document Processing**: Upload PDF, CSV, and text files for analysis
- **Advanced Retrieval**: Semantic search using FAISS vector store for relevant document retrieval
- **Fine-tuned Responses**: Control model parameters for customized outputs
- **Intuitive UI**: User-friendly interface for seamless interaction

## Architecture

The application follows a modular architecture with the following components:

### 1. Document Processing Pipeline

- Handles document uploads (PDF, CSV, TXT)
- Extracts text content using specialized loaders
- Splits documents semantically using HuggingFace embeddings
- Creates and manages FAISS vector stores for efficient retrieval

### 2. Question-Answering System

- Connects to Ollama for large language model capabilities
- Provides prompt engineering for improved responses
- Integrates retrieval with language generation
- Offers customizable generation parameters:
  - Temperature (creativity control)
  - Top-p sampling (diversity control)
  - Max tokens (response length control)
  - Retrieval context size (information breadth)

### 3. User Interface

- Built with Streamlit for a responsive experience
- Provides chat-based interaction
- Offers intuitive parameter controls
- Ensures clean document management

## Technical Implementation

### Vector Embeddings

The system uses HuggingFace embeddings to convert document text into semantic vector representations, allowing for contextual understanding and similarity-based retrieval.

### Semantic Chunking

Documents are split using semantic chunking to maintain coherent information units rather than arbitrary splits, improving retrieval quality.

### Vector Retrieval

FAISS (Facebook AI Similarity Search) provides efficient similarity search for large vector datasets, enabling fast and accurate document retrieval based on query relevance.

### Language Model Integration

The application connects to Ollama to access language models for response generation. Users can select from different models:
- qwen2.5-coder:3b
- qwen2.5:3b
- deepseek-r1:1.5b

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Or: Python 3.8+ and Ollama installed locally

### Docker Installation (Recommended)

The easiest way to run the application is using Docker:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/reason-about-data.git
   cd reason-about-data
   ```

2. Start the application:
   ```
   # On Linux/Mac
   ./start.sh
   
   # On Windows
   start.bat
   ```

3. Access the application in your browser:
   ```
   http://localhost:8501
   ```

This will launch:
- The Streamlit application
- Ollama model service
- Automatic model initialization

### Manual Installation

If you prefer to run without Docker:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/reason-about-data.git
   cd reason-about-data
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Start Ollama (if not already running):
   ```
   ollama serve
   ```

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage Guide

1. **Upload Documents**:
   - Navigate to the sidebar
   - Upload one or more PDF, CSV, or TXT files
   - Wait for processing to complete

2. **Configure Model Parameters**:
   - Adjust temperature (0.0-1.0) for creativity control
   - Set top-p sampling (0.1-1.0) for output diversity
   - Configure max tokens (64-2048) for response length
   - Set retrieval context size (1-10) for information retrieval

3. **Interact with Your Documents**:
   - Type questions in the chat input
   - Receive AI-generated responses based on document content
   - Continue the conversation with follow-up questions

## Docker Configuration

The Docker setup includes:

1. **Application Container**: Runs the Streamlit interface and RAG logic
2. **Ollama Container**: Provides the language model capabilities
3. **Init Container**: Automatically downloads required models on startup

Environment variables can be customized in the `.env` file:
- `STREAMLIT_SERVER_PORT`: Port for the web interface (default: 8501)
- `OLLAMA_HOST`: URL for Ollama service
- `DEFAULT_MODEL`: Default model to use for inference

Data persistence:
- Models are stored in a Docker volume for persistence between restarts
- Uploaded documents are stored in the `./data` directory

## Customization

### Adding New Model Support

To add additional Ollama models:
1. Update the `AVAILABLE_MODELS` list in `config/model.py`
2. Add the model to the initialization script in `scripts/init_models.sh`

### Supporting Additional File Types

Extend the `SUPPORTED_FILE_TYPES` list in `config/constants.py` and implement appropriate loaders in `services/document_service.py`.

## Troubleshooting

- **Processing Errors**: Check that your documents are properly formatted and not corrupted
- **Model Loading Issues**: Ensure Ollama is running and the selected models are installed
- **Docker Issues**: Check container logs with `docker-compose logs`
- **Performance Concerns**: For large documents, consider reducing the retrieval context size for faster responses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/) framework
- Vector search powered by [FAISS](https://github.com/facebookresearch/faiss)
- Language models provided by [Ollama](https://ollama.ai/)
- User interface created with [Streamlit](https://streamlit.io/) 