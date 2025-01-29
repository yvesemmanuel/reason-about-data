from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker


class DocumentProcessor:
    """A document processing pipeline for PDF files that creates search-optimized vector representations.

    The processor handles PDF loading, semantic text splitting, embedding generation, and vector store creation.
    Enables efficient similarity-based retrieval of document content.

    Typical usage:
        >>> processor = DocumentProcessor()
        >>> vector_store = processor.process_pdf("document.pdf")
        >>> retriever = processor.create_retriever(vector_store, top_k=5)

    Attributes:
        embedder: HuggingFace sentence transformer model for text embeddings
        text_splitter: Semantic chunker that maintains contextual coherence in splits
    """

    def __init__(self):
        """Initializes the document processor with default embedding model and text splitter.

        Uses:
        - HuggingFaceEmbeddings: Default 'sentence-transformers/all-mpnet-base-v2' model
        - SemanticChunker: Experimental chunker that splits documents based on semantic similarity
        """
        self.embedder = HuggingFaceEmbeddings()
        self.text_splitter = SemanticChunker(self.embedder)

    def process_pdf(self, file_path: str) -> FAISS:
        """Processes a PDF document into a search-optimized vector store.

        Pipeline:
        1. Load PDF content with text and layout preservation
        2. Split document using semantic-aware chunking
        3. Generate embeddings for each chunk
        4. Create FAISS index for efficient similarity search

        Parameters:
        -----------
            file_path (str): Path to PDF file (supports local, S3, and web URLs)

        Returns:
        --------
            FAISS: Vector store containing document chunks and their embeddings

        Raises:
        -------
            IOError: If the file cannot be loaded from the provided path
            ValueError: If the PDF contains no extractable text content
        """
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        documents = self.text_splitter.split_documents(docs)
        return FAISS.from_documents(documents, self.embedder)

    def create_retriever(
        self, vector_store: FAISS, top_k: int = 3
    ) -> VectorStoreRetriever:
        """Creates a configurable document retriever from a FAISS vector store.

        The retriever uses similarity search to find the most relevant document chunks
        based on input queries.

        Parameters:
        -----------
            vector_store (FAISS): Preprocessed vector store from process_pdf()
            top_k (int): Number of most similar documents to retrieve (default: 3)

        Returns:
        --------
            VectorStoreRetriever: Configured retriever instance ready for querying

        Example:
        --------
            >>> retriever = create_retriever(vector_store, top_k=5)
            >>> relevant_docs = retriever.invoke("What is machine learning?")
        """
        return vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
