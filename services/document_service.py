import tempfile
from typing import List, Dict, Any, Optional
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader, CSVLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.vectorstores.base import VectorStoreRetriever


class DocumentService:
    """Service for processing documents and creating vector stores."""

    def __init__(self):
        """Initialize document processor with embedding model and text splitter."""
        self.embedder = HuggingFaceEmbeddings()
        self.text_splitter = SemanticChunker(self.embedder)

    def process_files(self, uploaded_files) -> Optional[FAISS]:
        """Process multiple uploaded files and combine them into a single vector store.

        Args:
            uploaded_files: List of uploaded files from Streamlit

        Returns:
            Optional[FAISS]: Combined vector store or None if processing failed
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
                current_vector_store = self._process_single_file(
                    tmp_path, file_extension
                )

                if combined_vector_store is None:
                    combined_vector_store = current_vector_store
                else:
                    combined_vector_store.merge_from(current_vector_store)

            except Exception as e:
                if st:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                print(f"Error processing file: {str(e)}")
                continue

        return combined_vector_store

    def process_uploaded_paths(self, file_paths: List[str]) -> Optional[FAISS]:
        """Process multiple file paths and combine them into a single vector store.

        Args:
            file_paths: List of file paths to process

        Returns:
            Optional[FAISS]: Combined vector store or None if processing failed
        """
        combined_vector_store = None

        for file_path in file_paths:
            file_extension = file_path.split(".")[-1].lower()

            try:
                current_vector_store = self._process_single_file(
                    file_path, file_extension
                )

                if combined_vector_store is None:
                    combined_vector_store = current_vector_store
                else:
                    combined_vector_store.merge_from(current_vector_store)

            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                continue

        return combined_vector_store

    def _process_single_file(self, file_path: str, file_extension: str) -> FAISS:
        """Process a single file based on its extension.

        Args:
            file_path: Path to the file
            file_extension: Extension of the file (pdf, csv, txt)

        Returns:
            FAISS: Vector store created from the file

        Raises:
            ValueError: If file type is not supported
        """
        if file_extension == "pdf":
            return self._process_pdf(file_path)
        elif file_extension == "csv":
            return self._process_csv(file_path)
        elif file_extension == "txt":
            return self._process_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _create_vector_store(self, docs) -> FAISS:
        """Create a vector store from documents.

        Args:
            docs: Document objects

        Returns:
            FAISS: Vector store
        """
        return FAISS.from_documents(docs, self.embedder)

    def _process_pdf(self, file_path: str) -> FAISS:
        """Process a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            FAISS: Vector store created from the PDF
        """
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        return self._create_vector_store(split_docs)

    def _process_csv(self, file_path: str) -> FAISS:
        """Process a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            FAISS: Vector store created from the CSV
        """
        loader = CSVLoader(file_path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        return self._create_vector_store(split_docs)

    def _process_txt(self, file_path: str) -> FAISS:
        """Process a text file.

        Args:
            file_path: Path to the text file

        Returns:
            FAISS: Vector store created from the text file
        """
        loader = TextLoader(file_path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        return self._create_vector_store(split_docs)

    def create_retriever(self, vector_store: FAISS, top_k: int) -> VectorStoreRetriever:
        """Create a retriever from a vector store.

        Args:
            vector_store: Vector store to create a retriever from
            top_k: Number of documents to retrieve

        Returns:
            VectorStoreRetriever: Retriever
        """
        return vector_store.as_retriever(search_kwargs={"k": top_k})
