from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PDFPlumberLoader, CSVLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker


class DocumentProcessor:
    def __init__(self):
        self.embedder = HuggingFaceEmbeddings()
        self.text_splitter = SemanticChunker(self.embedder)

    def _create_vector_store(self, docs):
        return FAISS.from_documents(docs, self.embedder)

    def process_pdf(self, file_path: str) -> FAISS:
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        return self._create_vector_store(split_docs)

    def process_csv(self, file_path: str) -> FAISS:
        loader = CSVLoader(file_path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        return self._create_vector_store(split_docs)

    def process_txt(self, file_path: str) -> FAISS:
        loader = TextLoader(file_path)
        docs = loader.load()
        split_docs = self.text_splitter.split_documents(docs)
        return self._create_vector_store(split_docs)

    def create_retriever(self, vector_store: FAISS, top_k: int):
        return vector_store.as_retriever(search_kwargs={"k": top_k})
