from typing import Dict, Any

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables.base import Runnable

from config.model import OLLAMA_MODEL_ID


class QAService:
    """Service for building question answering chains."""

    def build_qa_chain(self, retriever: VectorStoreRetriever) -> Runnable:
        """Build a question answering chain.

        Args:
            retriever: Document retriever

        Returns:
            Runnable: Question answering chain
        """
        llm = self._create_llm()
        prompt_template = self._create_prompt_template()
        document_prompt = self._create_document_prompt()

        combine_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt_template,
            document_variable_name="context",
            document_prompt=document_prompt,
        )

        return create_retrieval_chain(
            combine_docs_chain=combine_chain, retriever=retriever
        )

    def _create_llm(self) -> OllamaLLM:
        """Create a language model.

        Returns:
            OllamaLLM: Language model
        """
        return OllamaLLM(model=OLLAMA_MODEL_ID)

    def _create_prompt_template(self) -> PromptTemplate:
        """Create a prompt template for the QA chain.

        Returns:
            PromptTemplate: Prompt template
        """
        return PromptTemplate.from_template(
            "1. Use context to answer the question\n"
            "2. If unsure, say 'I don't know'\n"
            "3. Keep answer concise (3-4 sentences)\n\n"
            "Context: {context}\n\nQuestion: {input}\n\nAnswer:"
        )

    def _create_document_prompt(self) -> PromptTemplate:
        """Create a document prompt for formatting context.

        Returns:
            PromptTemplate: Document prompt
        """
        return PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )
