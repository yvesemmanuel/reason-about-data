"""
Shared pytest fixtures for all tests.
"""

import pytest
from unittest.mock import Mock, MagicMock
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_huggingface import HuggingFaceEmbeddings


@pytest.fixture
def mock_ollama_llm():
    """Create a mocked Ollama LLM instance."""
    mock_llm = Mock(spec=OllamaLLM)
    mock_llm.invoke.return_value = "Mocked LLM response"
    return mock_llm


@pytest.fixture
def mock_prompt_template():
    """Create a mocked PromptTemplate instance."""
    mock_template = Mock(spec=PromptTemplate)
    mock_template.format.return_value = "Mocked prompt"
    return mock_template


@pytest.fixture
def mock_retriever():
    """Create a mocked VectorStoreRetriever instance."""
    mock_retriever = Mock(spec=VectorStoreRetriever)
    mock_retriever.get_relevant_documents.return_value = [
        MagicMock(page_content="Document content", metadata={"source": "test.pdf"})
    ]
    return mock_retriever


@pytest.fixture
def mock_vector_store():
    """Create a mocked FAISS vector store instance."""
    mock_vs = Mock(spec=FAISS)
    mock_retriever = Mock(spec=VectorStoreRetriever)
    mock_vs.as_retriever.return_value = mock_retriever
    mock_vs.merge_from = Mock()
    return mock_vs


@pytest.fixture
def mock_embeddings():
    """Create a mocked HuggingFaceEmbeddings instance."""
    mock_embeddings = Mock(spec=HuggingFaceEmbeddings)
    mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    return mock_embeddings
