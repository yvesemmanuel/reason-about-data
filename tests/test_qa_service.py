"""
Tests for the QA service module.

This module contains tests for the QAService class, including tests for building
question answering chains, creating language models, and querying the chains.
"""

import pytest
from unittest.mock import patch, Mock, MagicMock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from services.qa_service import QAService


class TestQAService:
    """Test suite for the QAService class."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.service = QAService()

    @patch("services.qa_service.create_stuff_documents_chain")
    @patch("services.qa_service.create_retrieval_chain")
    def test_build_qa_chain_success(
        self,
        mock_create_retrieval_chain,
        mock_create_stuff_chain,
        mock_retriever,
        mock_ollama_llm,
    ):
        """Test successful creation of QA chain."""
        mock_combine_chain = Mock()
        mock_qa_chain = Mock()

        mock_create_stuff_chain.return_value = (mock_combine_chain, None)
        mock_create_retrieval_chain.return_value = (mock_qa_chain, None)

        with patch.object(
            self.service, "_create_llm", return_value=(mock_ollama_llm, None)
        ) as mock_create_llm:
            with patch.object(
                self.service, "_create_prompt_template", return_value=(Mock(), None)
            ) as mock_create_prompt:
                with patch.object(
                    self.service, "_create_document_prompt", return_value=(Mock(), None)
                ) as mock_create_doc_prompt:
                    result, error = self.service.build_qa_chain(mock_retriever)

                    assert result == mock_qa_chain
                    assert error is None
                    mock_create_llm.assert_called_once()
                    mock_create_prompt.assert_called_once()
                    mock_create_doc_prompt.assert_called_once()
                    mock_create_stuff_chain.assert_called_once()
                    mock_create_retrieval_chain.assert_called_once_with(
                        mock_retriever, mock_combine_chain
                    )

    @patch("services.qa_service.create_stuff_documents_chain")
    def test_build_qa_chain_stuff_chain_error(
        self, mock_create_stuff_chain, mock_ollama_llm, mock_retriever
    ):
        """Test handling of error in stuff documents chain creation."""
        expected_error = Exception("Stuff chain creation error")
        mock_create_stuff_chain.return_value = (None, expected_error)

        with patch.object(
            self.service, "_create_llm", return_value=(mock_ollama_llm, None)
        ):
            with patch.object(
                self.service, "_create_prompt_template", return_value=(Mock(), None)
            ):
                with patch.object(
                    self.service, "_create_document_prompt", return_value=(Mock(), None)
                ):
                    result, error = self.service.build_qa_chain(mock_retriever)

                    assert result is None
                    assert error == expected_error

    def test_build_qa_chain_llm_error(self, mock_retriever):
        """Test handling of error in LLM creation."""
        expected_error = Exception("LLM creation error")

        with patch.object(
            self.service, "_create_llm", return_value=(None, expected_error)
        ):
            result, error = self.service.build_qa_chain(mock_retriever)

            assert result is None
            assert error == expected_error

    def test_build_qa_chain_prompt_template_error(
        self, mock_ollama_llm, mock_retriever
    ):
        """Test handling of error in prompt template creation."""
        expected_error = Exception("Prompt template creation error")

        with patch.object(
            self.service, "_create_llm", return_value=(mock_ollama_llm, None)
        ):
            with patch.object(
                self.service,
                "_create_prompt_template",
                return_value=(None, expected_error),
            ):
                result, error = self.service.build_qa_chain(mock_retriever)

                assert result is None
                assert error == expected_error

    def test_build_qa_chain_document_prompt_error(
        self, mock_ollama_llm, mock_prompt_template, mock_retriever
    ):
        """Test handling of error in document prompt creation."""
        expected_error = Exception("Document prompt creation error")

        with patch.object(
            self.service, "_create_llm", return_value=(mock_ollama_llm, None)
        ):
            with patch.object(
                self.service,
                "_create_prompt_template",
                return_value=(mock_prompt_template, None),
            ):
                with patch.object(
                    self.service,
                    "_create_document_prompt",
                    return_value=(None, expected_error),
                ):
                    result, error = self.service.build_qa_chain(mock_retriever)

                    assert result is None
                    assert error == expected_error

    @patch("services.qa_service.create_retrieval_chain")
    def test_build_qa_chain_retrieval_chain_error(
        self,
        mock_create_retrieval_chain,
        mock_ollama_llm,
        mock_prompt_template,
        mock_retriever,
    ):
        """Test handling of error in retrieval chain creation."""
        expected_error = Exception("Retrieval chain creation error")
        mock_create_retrieval_chain.return_value = (None, expected_error)

        with patch.object(
            self.service, "_create_llm", return_value=(mock_ollama_llm, None)
        ):
            with patch.object(
                self.service,
                "_create_prompt_template",
                return_value=(mock_prompt_template, None),
            ):
                with patch.object(
                    self.service, "_create_document_prompt", return_value=(Mock(), None)
                ):
                    with patch(
                        "services.qa_service.create_stuff_documents_chain",
                        return_value=(Mock(), None),
                    ):
                        result, error = self.service.build_qa_chain(mock_retriever)

                        assert result is None
                        assert error == expected_error

    def test_build_direct_chain_success(self, mock_ollama_llm):
        """Test successful creation of direct QA chain."""
        mock_prompt = Mock(spec=PromptTemplate)

        with patch.object(
            self.service, "_create_llm", return_value=(mock_ollama_llm, None)
        ):
            with patch.object(
                self.service, "_create_direct_prompt", return_value=mock_prompt
            ):
                with patch("services.qa_service.LLMChain") as mock_llm_chain:
                    chain_instance = Mock(spec=LLMChain)
                    mock_llm_chain.return_value = chain_instance

                    result, error = self.service.build_direct_chain()

                    assert result == chain_instance
                    assert error is None
                    mock_llm_chain.assert_called_once_with(
                        llm=mock_ollama_llm, prompt=mock_prompt, output_key="answer"
                    )

    def test_build_direct_chain_llm_error(self):
        """Test handling of error in LLM creation for direct chain."""
        expected_error = Exception("LLM creation error")

        with patch.object(
            self.service, "_create_llm", return_value=(None, expected_error)
        ):
            result, error = self.service.build_direct_chain()

            assert result is None
            assert error == expected_error

    def test_query_direct_success(self):
        """Test successful query of direct QA chain."""
        mock_chain = Mock(spec=LLMChain)
        expected_result = {"answer": "Test answer"}
        mock_chain.invoke.return_value = expected_result

        result, error = self.service.query_direct(mock_chain, "Test query")

        assert result == expected_result
        assert error is None
        mock_chain.invoke.assert_called_once_with(
            {"query": "Test query", "chat_history": ""}
        )

    def test_query_direct_with_history(self):
        """Test query of direct QA chain with chat history."""
        mock_chain = Mock(spec=LLMChain)
        expected_result = {"answer": "Test answer with history"}
        mock_chain.invoke.return_value = expected_result

        result, error = self.service.query_direct(
            mock_chain, "Test query", "Previous chat history"
        )

        assert result == expected_result
        assert error is None
        mock_chain.invoke.assert_called_once_with(
            {"query": "Test query", "chat_history": "Previous chat history"}
        )

    def test_query_direct_error(self):
        """Test handling of error in direct chain query."""
        mock_chain = Mock(spec=LLMChain)
        expected_error = Exception("Chain invoke error")
        mock_chain.invoke.side_effect = expected_error

        result, error = self.service.query_direct(mock_chain, "Test query")

        assert result is None
        assert error == expected_error

    @patch("services.qa_service.OllamaLLM")
    @patch("services.qa_service.OLLAMA_HOST", "http://localhost:11434")
    @patch("services.qa_service.DEFAULT_MODEL", "test-model")
    def test_create_llm_success(self, mock_ollama_class):
        """Test successful creation of Ollama LLM."""
        mock_llm_instance = Mock()
        mock_ollama_class.return_value = mock_llm_instance

        result, error = self.service._create_llm()

        assert result == mock_llm_instance
        assert error is None
        mock_ollama_class.assert_called_once_with(
            model="test-model",
            base_url="http://localhost:11434",
            temperature=0.7,
            num_ctx=4096,
            num_predict=512,
            top_p=0.9,
        )

    @patch("services.qa_service.OllamaLLM")
    @patch("services.qa_service.AVAILABLE_MODELS", ["model1", "model2"])
    def test_create_llm_with_custom_params(self, mock_ollama_class):
        """Test LLM creation with custom parameters."""
        mock_llm_instance = Mock()
        mock_ollama_class.return_value = mock_llm_instance

        result, error = self.service._create_llm(
            model_id="model1", temperature=0.5, max_tokens=256, top_p=0.8
        )

        assert result == mock_llm_instance
        assert error is None
        mock_ollama_class.assert_called_once()
        assert mock_ollama_class.call_args.kwargs["temperature"] == 0.5
        assert mock_ollama_class.call_args.kwargs["num_predict"] == 256
        assert mock_ollama_class.call_args.kwargs["top_p"] == 0.8

    @patch("services.qa_service.OllamaLLM")
    def test_create_llm_error(self, mock_ollama_class):
        """Test handling of error in Ollama LLM creation."""
        expected_error = Exception("OllamaLLM creation error")
        mock_ollama_class.side_effect = expected_error

        result, error = self.service._create_llm()

        assert result is None
        assert error == expected_error

    @patch("services.qa_service.PromptTemplate")
    def test_create_prompt_template_success(self, mock_prompt_template_class):
        """Test successful creation of prompt template."""
        mock_template = Mock()
        mock_prompt_template_class.from_template.return_value = mock_template

        result, error = self.service._create_prompt_template()

        assert result == mock_template
        assert error is None
        mock_prompt_template_class.from_template.assert_called_once()

    @patch("services.qa_service.PromptTemplate")
    def test_create_prompt_template_error(self, mock_prompt_template_class):
        """Test handling of error in prompt template creation."""
        expected_error = Exception("PromptTemplate creation error")
        mock_prompt_template_class.from_template.side_effect = expected_error

        result, error = self.service._create_prompt_template()

        assert result is None
        assert error == expected_error

    @patch("services.qa_service.PromptTemplate")
    def test_create_direct_prompt(self, mock_prompt_template_class):
        """Test creation of direct prompt template."""
        mock_template = Mock()
        mock_prompt_template_class.return_value = mock_template

        result = self.service._create_direct_prompt()

        assert result == mock_template
        mock_prompt_template_class.assert_called_once()
        assert "query" in mock_prompt_template_class.call_args.kwargs["input_variables"]
        assert (
            "chat_history"
            in mock_prompt_template_class.call_args.kwargs["input_variables"]
        )

    @patch("services.qa_service.PromptTemplate")
    def test_create_document_prompt_success(self, mock_prompt_template_class):
        """Test successful creation of document prompt template."""
        mock_template = Mock()
        mock_prompt_template_class.return_value = mock_template

        result, error = self.service._create_document_prompt()

        assert result == mock_template
        assert error is None
        mock_prompt_template_class.assert_called_once()
        assert (
            "document_content"
            in mock_prompt_template_class.call_args.kwargs["input_variables"]
        )
        assert (
            "metadata" in mock_prompt_template_class.call_args.kwargs["input_variables"]
        )

    @patch("services.qa_service.PromptTemplate")
    def test_create_document_prompt_error(self, mock_prompt_template_class):
        """Test handling of error in document prompt template creation."""
        expected_error = Exception("Document PromptTemplate creation error")
        mock_prompt_template_class.side_effect = expected_error

        result, error = self.service._create_document_prompt()

        assert result is None
        assert error == expected_error
