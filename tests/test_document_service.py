"""
Tests for the Document service module.

This module contains tests for the DocumentService class, including tests for
processing files, creating vector stores, and creating document retrievers.
"""

from unittest.mock import patch, Mock, MagicMock

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores.base import VectorStoreRetriever

from services.document_service import DocumentService


class TestDocumentService:
    """Test suite for the DocumentService class."""

    def setup_method(self):
        """Set up test environment before each test method."""
        self.service = DocumentService()

    @patch("services.document_service.HuggingFaceEmbeddings")
    @patch("services.document_service.SemanticChunker")
    def test_init(self, mock_chunker_class, mock_embeddings_class):
        """Test initialization of DocumentService."""
        mock_embeddings = Mock()
        mock_chunker = Mock()
        mock_embeddings_class.return_value = mock_embeddings
        mock_chunker_class.return_value = mock_chunker

        service = DocumentService()

        assert service.embedder == mock_embeddings
        assert service.text_splitter == mock_chunker
        mock_chunker_class.assert_called_once_with(mock_embeddings)

    @patch("services.document_service.tempfile.NamedTemporaryFile")
    def test_process_files_success(self, mock_temp_file):
        """Test successful processing of multiple files."""
        pdf_file = MagicMock()
        pdf_file.name = "test.pdf"
        pdf_file.getvalue.return_value = b"PDF content"

        csv_file = MagicMock()
        csv_file.name = "test.csv"
        csv_file.getvalue.return_value = b"CSV content"

        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = mock_temp
        mock_temp.name = "temp_file_path"
        mock_temp_file.return_value = mock_temp

        mock_pdf_store = Mock(spec=FAISS)
        mock_csv_store = Mock(spec=FAISS)

        with patch.object(
            self.service,
            "_process_single_file",
            side_effect=[(mock_pdf_store, None), (mock_csv_store, None)],
        ) as mock_process_single:
            result, error = self.service.process_files([pdf_file, csv_file])

            assert error is None
            assert mock_process_single.call_count == 2
            mock_process_single.assert_any_call("temp_file_path", "pdf")
            mock_process_single.assert_any_call("temp_file_path", "csv")

    @patch("services.document_service.tempfile.NamedTemporaryFile")
    def test_process_files_error(self, mock_temp_file):
        """Test handling of error in processing files."""
        pdf_file = MagicMock()
        pdf_file.name = "test.pdf"
        pdf_file.getvalue.return_value = b"PDF content"

        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = mock_temp
        mock_temp.name = "temp_file_path"
        mock_temp_file.return_value = mock_temp

        expected_error = Exception("Processing error")

        with patch.object(
            self.service, "_process_single_file", return_value=(None, expected_error)
        ) as mock_process_single:
            result, error = self.service.process_files([pdf_file])

            assert result is None
            assert error == expected_error
            mock_process_single.assert_called_once_with("temp_file_path", "pdf")

    @patch("services.document_service.tempfile.NamedTemporaryFile")
    def test_process_files_exception(self, mock_temp_file):
        """Test handling of exception in processing files."""
        pdf_file = MagicMock()
        pdf_file.name = "test.pdf"
        pdf_file.getvalue.return_value = b"PDF content"

        mock_temp = MagicMock()
        mock_temp.__enter__.return_value = mock_temp
        mock_temp.name = "temp_file_path"
        mock_temp_file.return_value = mock_temp

        expected_exception = Exception("Unexpected error")

        with patch.object(
            self.service, "_process_single_file", side_effect=expected_exception
        ) as mock_process_single:
            result, error = self.service.process_files([pdf_file])

            assert result is None
            assert error == expected_exception
            mock_process_single.assert_called_once_with("temp_file_path", "pdf")

    def test_process_uploaded_paths_success(self):
        """Test successful processing of multiple file paths."""
        file_paths = ["test.pdf", "test.csv", "test.txt"]

        mock_pdf_store = Mock(spec=FAISS)
        mock_csv_store = Mock(spec=FAISS)
        mock_txt_store = Mock(spec=FAISS)
        mock_combined_store = Mock(spec=FAISS)

        mock_pdf_store.merge_from = Mock()
        mock_pdf_store.merge_from.return_value = mock_combined_store

        with patch.object(
            self.service,
            "_process_single_file",
            side_effect=[
                (mock_pdf_store, None),
                (mock_csv_store, None),
                (mock_txt_store, None),
            ],
        ) as mock_process_single:
            result, error = self.service.process_uploaded_paths(file_paths)

            assert result == mock_pdf_store
            assert error is None
            assert mock_process_single.call_count == 3
            mock_process_single.assert_any_call("test.pdf", "pdf")
            mock_process_single.assert_any_call("test.csv", "csv")
            mock_process_single.assert_any_call("test.txt", "txt")
            
            from unittest.mock import call
            mock_pdf_store.merge_from.assert_has_calls([
                call(mock_csv_store),
                call(mock_txt_store)
            ])

    def test_process_uploaded_paths_error(self):
        """Test handling of error in processing file paths."""
        file_path = ["test.pdf"]

        expected_error = Exception("Processing error")

        with patch.object(
            self.service, "_process_single_file", return_value=(None, expected_error)
        ) as mock_process_single:
            result, error = self.service.process_uploaded_paths(file_path)

            assert result is None
            assert error == expected_error
            mock_process_single.assert_called_once_with("test.pdf", "pdf")

    def test_process_uploaded_paths_exception(self):
        """Test handling of exception in processing file paths."""
        file_path = ["test.pdf"]

        expected_exception = Exception("Unexpected error")

        with patch.object(
            self.service, "_process_single_file", side_effect=expected_exception
        ) as mock_process_single:
            result, error = self.service.process_uploaded_paths(file_path)

            assert result is None
            assert error == expected_exception
            mock_process_single.assert_called_once_with("test.pdf", "pdf")

    def test_process_single_file_supported_extension(self):
        """Test processing single file with supported extension."""
        mock_pdf_store = Mock(spec=FAISS)

        with patch.object(
            self.service, "_process_pdf", return_value=(mock_pdf_store, None)
        ) as mock_process_pdf:
            with patch.object(
                self.service, "_process_csv", return_value=(None, None)
            ) as mock_process_csv:
                with patch.object(
                    self.service, "_process_txt", return_value=(None, None)
                ) as mock_process_txt:
                    result, error = self.service._process_single_file("test.pdf", "pdf")

                    assert result == mock_pdf_store
                    assert error is None
                    mock_process_pdf.assert_called_once_with("test.pdf")
                    mock_process_csv.assert_not_called()
                    mock_process_txt.assert_not_called()

                    mock_process_pdf.reset_mock()

                    mock_csv_store = Mock(spec=FAISS)
                    mock_process_csv.return_value = (mock_csv_store, None)
                    result, error = self.service._process_single_file("test.csv", "csv")

                    assert result == mock_csv_store
                    assert error is None
                    mock_process_pdf.assert_not_called()
                    mock_process_csv.assert_called_once_with("test.csv")
                    mock_process_txt.assert_not_called()

                    mock_process_csv.reset_mock()

                    mock_txt_store = Mock(spec=FAISS)
                    mock_process_txt.return_value = (mock_txt_store, None)
                    result, error = self.service._process_single_file("test.txt", "txt")

                    assert result == mock_txt_store
                    assert error is None
                    mock_process_pdf.assert_not_called()
                    mock_process_csv.assert_not_called()
                    mock_process_txt.assert_called_once_with("test.txt")

    def test_process_single_file_unsupported_extension(self):
        """Test processing single file with unsupported extension."""
        result, error = self.service._process_single_file("test.docx", "docx")

        assert result is None
        assert isinstance(error, ValueError)
        assert "Unsupported file type: docx" in str(error)

    def test_create_vector_store_success(self, mock_embeddings):
        """Test successful creation of vector store."""
        docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]
        mock_vs = Mock(spec=FAISS)

        with patch.object(
            self.service.text_splitter, "split_documents", return_value=docs
        ) as mock_split:
            with patch(
                "services.document_service.FAISS.from_documents", return_value=mock_vs
            ) as mock_faiss:
                result, error = self.service._create_vector_store(docs)

                assert result == mock_vs
                assert error is None
                mock_split.assert_called_once_with(docs)
                mock_faiss.assert_called_once_with(docs, self.service.embedder)

    def test_create_vector_store_error(self, mock_embeddings):
        """Test handling of error in vector store creation."""
        docs = [Document(page_content="Test content", metadata={"source": "test.pdf"})]
        expected_error = Exception("Vector store creation error")

        with patch.object(
            self.service.text_splitter, "split_documents", side_effect=expected_error
        ) as mock_split:
            result, error = self.service._create_vector_store(docs)

            assert result is None
            assert error == expected_error
            mock_split.assert_called_once_with(docs)

    @patch("services.document_service.PDFPlumberLoader")
    def test_process_pdf_success(self, mock_loader_class):
        """Test successful processing of PDF file."""
        mock_loader = Mock()
        mock_docs = [
            Document(page_content="PDF content", metadata={"source": "test.pdf"})
        ]
        mock_loader.load.return_value = mock_docs
        mock_loader_class.return_value = mock_loader

        mock_vs = Mock(spec=FAISS)

        with patch.object(
            self.service, "_create_vector_store", return_value=(mock_vs, None)
        ) as mock_create_vs:
            result, error = self.service._process_pdf("test.pdf")

            assert result == mock_vs
            assert error is None
            mock_loader_class.assert_called_once_with("test.pdf")
            mock_loader.load.assert_called_once()
            mock_create_vs.assert_called_once_with(mock_docs)

    @patch("services.document_service.PDFPlumberLoader")
    def test_process_pdf_error(self, mock_loader_class):
        """Test handling of error in PDF processing."""
        mock_loader = Mock()
        expected_error = Exception("PDF loading error")
        mock_loader.load.side_effect = expected_error
        mock_loader_class.return_value = mock_loader

        result, error = self.service._process_pdf("test.pdf")

        assert result is None
        assert error == expected_error
        mock_loader_class.assert_called_once_with("test.pdf")
        mock_loader.load.assert_called_once()

    @patch("services.document_service.CSVLoader")
    def test_process_csv_success(self, mock_loader_class):
        """Test successful processing of CSV file."""
        mock_loader = Mock()
        mock_docs = [
            Document(page_content="CSV content", metadata={"source": "test.csv"})
        ]
        mock_loader.load.return_value = mock_docs
        mock_loader_class.return_value = mock_loader

        mock_vs = Mock(spec=FAISS)

        with patch.object(
            self.service, "_create_vector_store", return_value=(mock_vs, None)
        ) as mock_create_vs:
            result, error = self.service._process_csv("test.csv")

            assert result == mock_vs
            assert error is None
            mock_loader_class.assert_called_once_with("test.csv")
            mock_loader.load.assert_called_once()
            mock_create_vs.assert_called_once_with(mock_docs)

    @patch("services.document_service.TextLoader")
    def test_process_txt_success(self, mock_loader_class):
        """Test successful processing of TXT file."""
        mock_loader = Mock()
        mock_docs = [
            Document(page_content="Text content", metadata={"source": "test.txt"})
        ]
        mock_loader.load.return_value = mock_docs
        mock_loader_class.return_value = mock_loader

        mock_vs = Mock(spec=FAISS)

        with patch.object(
            self.service, "_create_vector_store", return_value=(mock_vs, None)
        ) as mock_create_vs:
            result, error = self.service._process_txt("test.txt")

            assert result == mock_vs
            assert error is None
            mock_loader_class.assert_called_once_with("test.txt")
            mock_loader.load.assert_called_once()
            mock_create_vs.assert_called_once_with(mock_docs)

    def test_create_retriever_success(self, mock_vector_store):
        """Test successful creation of document retriever."""
        mock_retriever = Mock(spec=VectorStoreRetriever)
        mock_vector_store.as_retriever.return_value = mock_retriever

        result, error = self.service.create_retriever(mock_vector_store, 5)

        assert result == mock_retriever
        assert error is None
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})

    def test_create_retriever_error(self, mock_vector_store):
        """Test handling of error in retriever creation."""
        expected_error = Exception("Retriever creation error")
        mock_vector_store.as_retriever.side_effect = expected_error

        result, error = self.service.create_retriever(mock_vector_store, 5)

        assert result is None
        assert error == expected_error
        mock_vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
