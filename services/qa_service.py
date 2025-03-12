from typing import Optional
import os

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables.base import Runnable

from config.model import OLLAMA_MODEL_ID, AVAILABLE_MODELS


class QAService:
    """Service for building question answering chains."""

    def build_qa_chain(
        self,
        retriever: VectorStoreRetriever,
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> Runnable:
        """Build a question answering chain.

        Parameters:
        -----------
            retriever: Document retriever
            model_id: Optional model ID to use (defaults to configured model)
            temperature: Controls randomness in output generation (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter (nucleus sampling)

        Returns:
        --------
            Runnable: Question answering chain
        """
        llm = self._create_llm(model_id, temperature, max_tokens, top_p)
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

    def _create_llm(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> OllamaLLM:
        """Create a language model.

        Parameters:
        -----------
            model_id: Optional model ID to use (defaults to configured model)
            temperature: Controls randomness in output generation (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter (nucleus sampling)

        Returns:
        --------
            OllamaLLM: Language model
        """
        # Get Ollama host from environment variable or use default
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

        if model_id in AVAILABLE_MODELS:
            return OllamaLLM(
                model=model_id,
                temperature=temperature,
                num_predict=max_tokens,
                top_p=top_p,
                base_url=ollama_host,
            )

        return OllamaLLM(
            model=OLLAMA_MODEL_ID,
            temperature=temperature,
            num_predict=max_tokens,
            top_p=top_p,
            base_url=ollama_host,
        )

    def _create_prompt_template(self) -> PromptTemplate:
        """Create a prompt template for the QA chain.

        Parameters:
        -----------
            None

        Returns:
        --------
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

        Parameters:
        -----------
            None

        Returns:
        --------
            PromptTemplate: Document prompt
        """
        return PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )
