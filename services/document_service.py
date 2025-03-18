"""Document service for processing documents and creating vector stores.

This module provides functionality for processing uploaded documents and
creating vector stores for efficient retrieval.
"""

import tempfile
from typing import List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    CSVLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.vectorstores.base import VectorStoreRetriever


class DocumentService:
    """Service for processing documents and creating vector stores."""

    def __init__(self):
        """Initialize document processor with embedding model and text splitter."""
        self.embedder = HuggingFaceEmbeddings()
        self.text_splitter = SemanticChunker(self.embedder)

    def process_files(
        self, uploaded_files
    ) -> Tuple[Optional[FAISS], Optional[Exception]]:
        """Process multiple uploaded files and combine them into a single vector store.

        Parameters:
        -----------
        uploaded_files: List[UploadedFile]
            List of uploaded files from Streamlit

        Returns:
        --------
        Tuple[Optional[FAISS], Optional[Exception]]: Combined vector store and error
        """
        combined_vector_store = None

        for uploaded_file in uploaded_files:
            file_extension = uploaded_file.name.split(".")[-1].lower()

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_extension}"
            ) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                current_vector_store, error = self._process_single_file(
                    tmp_path, file_extension
                )

                if error:
                    return None, error

                if combined_vector_store is None:
                    combined_vector_store = current_vector_store
                else:
                    combined_vector_store.merge_from(current_vector_store)
            except Exception as e:
                return None, e

        return None, None

    def process_uploaded_paths(
        self, file_paths: List[str]
    ) -> Tuple[Optional[FAISS], Optional[Exception]]:
        """Process multiple file paths and combine them into a single vector store.

        Parameters:
        -----------
        file_paths: List[str]
            List of paths to files

        Returns:
        --------
        Tuple[Optional[FAISS], Optional[Exception]]: Combined vector store and error
        """
        combined_vector_store = None

        for file_path in file_paths:
            file_extension = file_path.split(".")[-1].lower()

            try:
                current_vector_store, error = self._process_single_file(
                    file_path, file_extension
                )

                if error:
                    return None, error

                if combined_vector_store is None:
                    combined_vector_store = current_vector_store
                else:
                    combined_vector_store.merge_from(current_vector_store)
            except Exception as e:
                return None, e

        return combined_vector_store, None

    def _process_single_file(
        self, file_path: str, file_extension: str
    ) -> Tuple[Optional[FAISS], Optional[Exception]]:
        """Process a single file and create a vector store.

        Parameters:
        -----------
        file_path: str
            Path to the file
        file_extension: str
            Extension of the file

        Returns:
        --------
        Tuple[Optional[FAISS], Optional[Exception]]: Vector store and error

        Raises:
        -------
        ValueError: If the file type is not supported
        """
        if file_extension == "pdf":
            return self._process_pdf(file_path)
        elif file_extension == "csv":
            return self._process_csv(file_path)
        elif file_extension == "txt":
            return self._process_txt(file_path)
        else:
            return None, ValueError(f"Unsupported file type: {file_extension}")

    def _create_vector_store(self, docs) -> Tuple[Optional[FAISS], Optional[Exception]]:
        """Create a vector store from documents.

        Parameters:
        -----------
        docs: List[Document]
            List of documents

        Returns:
        --------
        Tuple[Optional[FAISS], Optional[Exception]]: Vector store and error
        """
        try:
            chunks = self.text_splitter.split_documents(docs)
            return FAISS.from_documents(chunks, self.embedder), None
        except Exception as e:
            return None, e

    def _process_pdf(
        self, file_path: str
    ) -> Tuple[Optional[FAISS], Optional[Exception]]:
        """Process a PDF file.

        Parameters:
        -----------
        file_path: str
            Path to the PDF file

        Returns:
        --------
        Tuple[Optional[FAISS], Optional[Exception]]: Vector store and error
        """
        try:
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()

            return self._create_vector_store(documents)
        except Exception as e:
            return None, e

    def _process_csv(
        self, file_path: str
    ) -> Tuple[Optional[FAISS], Optional[Exception]]:
        """Process a CSV file.

        Parameters:
        -----------
        file_path: str
            Path to the CSV file

        Returns:
        --------
        Tuple[Optional[FAISS], Optional[Exception]]: Vector store and error
        """
        try:
            loader = CSVLoader(file_path)
            documents = loader.load()

            return self._create_vector_store(documents)
        except Exception as e:
            return None, e

    def _process_txt(
        self, file_path: str
    ) -> Tuple[Optional[FAISS], Optional[Exception]]:
        """Process a text file.

        Parameters:
        -----------
        file_path: str
            Path to the text file

        Returns:
        --------
        Tuple[Optional[FAISS], Optional[Exception]]: Vector store and error
        """
        try:
            loader = TextLoader(file_path)
            documents = loader.load()

            return self._create_vector_store(documents)
        except Exception as e:
            return None, e

    def create_retriever(
        self, vector_store: FAISS, top_k: int
    ) -> Tuple[VectorStoreRetriever, Optional[Exception]]:
        """Create a document retriever from a vector store.

        Parameters:
        -----------
        vector_store: FAISS
            Vector store to retrieve from
        top_k: int
            Number of documents to retrieve

        Returns:
        --------
        Tuple[VectorStoreRetriever, Optional[Exception]]: Document retriever and error
        """
        try:
            return vector_store.as_retriever(search_kwargs={"k": top_k}), None
        except Exception as e:
            return None, e
