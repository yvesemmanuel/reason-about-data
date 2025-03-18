"""
This module provides a service for building question answering chains.

It includes methods for building retrieval-based and direct QA chains,
querying the chains, and creating language model instances.
"""

from typing import Optional, Dict, Any, Tuple

from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables.base import Runnable
from langchain.chains import LLMChain

from config.model import AVAILABLE_MODELS
from config.settings import OLLAMA_HOST, DEFAULT_MODEL


class QAService:
    """Service for building question answering chains."""

    def build_qa_chain(
        self,
        retriever: VectorStoreRetriever,
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> Tuple[Optional[Runnable], Optional[Exception]]:
        """Build a question answering chain with retrieval.

        Parameters:
        -----------
        retriever: VectorStoreRetriever
            Document retriever
        model_id: Optional[str]
            Optional model ID to use (defaults to configured model)
        temperature: float
            Controls randomness in output generation (0.0-1.0)
        max_tokens: int
            Maximum number of tokens to generate
        top_p: float
            Top-p sampling parameter (nucleus sampling)

        Returns:
        --------
        Tuple[Optional[Runnable], Optional[Exception]]: Question answering chain and error
        """
        try:
            llm, error = self._create_llm(model_id, temperature, max_tokens, top_p)
            if error:
                return None, error

            prompt_template, error = self._create_prompt_template()
            if error:
                return None, error

            document_prompt, error = self._create_document_prompt()
            if error:
                return None, error

            combine_chain, error = create_stuff_documents_chain(
                llm=llm,
                prompt=prompt_template,
                document_prompt=document_prompt,
            )
            if error:
                return None, error

            qa_chain, error = create_retrieval_chain(retriever, combine_chain)
            if error:
                return None, error

            return qa_chain, None
        except Exception as e:
            return None, e

    def build_direct_chain(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> Tuple[Optional[LLMChain], Optional[Exception]]:
        """Build a direct question answering chain without retrieval.

        Parameters:
        -----------
        model_id: Optional[str]
            Optional model ID to use (defaults to configured model)
        temperature: float
            Controls randomness in output generation (0.0-1.0)
        max_tokens: int
            Maximum number of tokens to generate
        top_p: float
            Top-p sampling parameter (nucleus sampling)

        Returns:
        --------
        Tuple[Optional[LLMChain], Optional[Exception]]: Direct question answering chain and error
        """
        try:
            llm, error = self._create_llm(model_id, temperature, max_tokens, top_p)
            if error:
                return None, error

            prompt_template = self._create_direct_prompt()

            direct_chain = llm | prompt_template
            return direct_chain, None
        except Exception as e:
            return None, e

    def query_direct(
        self, chain: LLMChain, query: str, chat_history: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Exception]]:
        """Query the direct QA chain.

        Parameters:
        -----------
        chain: LLMChain
            The direct QA chain
        query: str
            The user's query
        chat_history: Optional[str]
            Optional chat history as a string

        Returns:
        --------
        Tuple[Optional[Dict[str, Any]], Optional[Exception]]: Dictionary containing the answer and error
        """
        try:
            if chat_history:
                return chain.invoke(
                    {"query": query, "chat_history": chat_history}
                ), None
            else:
                return chain.invoke({"query": query, "chat_history": ""}), None
        except Exception as e:
            return None, e

    def _create_llm(
        self,
        model_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> Tuple[Optional[OllamaLLM], Optional[Exception]]:
        """Create an Ollama language model instance.

        Parameters:
        -----------
        model_id: Optional[str]
            Optional model ID to use (defaults to configured model)
        temperature: float
            Controls randomness in output generation (0.0-1.0)
        max_tokens: int
            Maximum number of tokens to generate
        top_p: float
            Top-p sampling parameter (nucleus sampling)

        Returns:
        --------
        Tuple[Optional[OllamaLLM], Optional[Exception]]: Configured language model and error
        """
        try:
            if not model_id or model_id not in AVAILABLE_MODELS:
                model_id = DEFAULT_MODEL

            ollama_host = OLLAMA_HOST

            return (
                OllamaLLM(
                    model=model_id,
                    base_url=ollama_host,
                    temperature=temperature,
                    num_ctx=4096,
                    num_predict=max_tokens,
                    top_p=top_p,
                ),
                None,
            )
        except Exception as e:
            return None, e

    def _create_prompt_template(
        self,
    ) -> Tuple[Optional[PromptTemplate], Optional[Exception]]:
        """Create a prompt template for the retrieval-based QA chain.

        Returns:
        --------
        Tuple[Optional[PromptTemplate], Optional[Exception]]: The prompt template and error
        """
        template = """
        <|im_start|>system
        You are a helpful, accurate, and friendly assistant. Use the following pieces of context to answer the user's question. If you don't know the answer or if the answer is not in the context, say "I'm sorry, I don't have enough information to answer that question." and suggest that the user try refining their query. Always be truthful and prioritize accuracy over giving a confident but potentially incorrect answer. Format your answers with Markdown to highlight key points or code when appropriate. If the context contains code, include relevant code examples and explanations in your answer.

        Context:
        {context}
        <|im_end|>

        <|im_start|>user
        {input}
        <|im_end|>

        <|im_start|>assistant
        """
        try:
            return PromptTemplate.from_template(template), None
        except Exception as e:
            return None, e

    def _create_direct_prompt(self) -> PromptTemplate:
        """Create a prompt template for the direct QA chain.

        Returns:
        --------
        PromptTemplate: The prompt template
        """
        template = """
        <|im_start|>system
        You are a helpful, accurate, and friendly assistant. Respond to the user's query to the best of your ability, providing comprehensive and detailed answers. Format your answers with Markdown to highlight key points or code when appropriate. If the user asks for code, provide complete and working examples along with explanations.

        Chat History:
        {chat_history}
        <|im_end|>

        <|im_start|>user
        {query}
        <|im_end|>

        <|im_start|>assistant
        """

        return PromptTemplate(
            input_variables=["query", "chat_history"], template=template
        )

    def _create_document_prompt(
        self,
    ) -> Tuple[Optional[PromptTemplate], Optional[Exception]]:
        """Create a document format prompt template.

        Returns:
        --------
        Tuple[Optional[PromptTemplate], Optional[Exception]]: The document format prompt template and error
        """
        try:
            return PromptTemplate(
                input_variables=["document_content", "metadata"],
                template="Content: {document_content}\nMetadata: {metadata}\n---\n",
            ), None
        except Exception as e:
            return None, e
