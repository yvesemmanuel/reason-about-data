from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama.llms import OllamaLLM
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.runnables.base import Runnable


class QAChainBuilder:
    """Builder class for constructing retrieval-augmented question answering chains.

    Provides a standardized pipeline for creating QA systems that:
    1. Retrieve relevant context documents
    2. Format documents with metadata
    3. Generate answers using specified LLM
    4. Enforce answer quality constraints

    Typical usage:
        >>> retriever = get_initialized_retriever()
        >>> qa_chain = QAChainBuilder.build_chain(retriever)
        >>> response = qa_chain.invoke({"input": "What is AI?"})
    """

    @staticmethod
    def build_chain(retriever: VectorStoreRetriever) -> Runnable:
        """Constructs an end-to-end retrieval-augmented QA pipeline.

        Architecture:
        1. Document Retrieval: Finds relevant context using vector similarity
        2. Context Formatting: Structures documents with content and source metadata
        3. Answer Generation: Uses LLM to synthesize answer from retrieved context

        Parameters:
        -----------
            retriever (VectorStoreRetriever): Initialized document retriever that
                implements similarity search. Should return documents with:
                - page_content: Text content of document chunk
                - source: Origin metadata for citation

        Returns:
        --------
            Runnable: Configured QA chain that accepts {"input": "question"} and
                returns {"answer": "generated_response", "context": [...]}

        Raises:
        -------
            ValueError: If input documents lack required metadata fields
            ConnectionError: If Ollama service is unavailable

        Example:
            >>> chain = QAChainBuilder.build_chain(retriever)
            >>> result = chain.invoke({"input": "What causes climate change?"})
            >>> print(result["answer"])
        """
        llm = OllamaLLM(model="deepseek-r1:1.5b")
        prompt_template = PromptTemplate.from_template(
            "1. Use context to answer the question\n"
            "2. If unsure, say 'I don't know'\n"
            "3. Keep answer concise (3-4 sentences)\n\n"
            "Context: {context}\n\nQuestion: {input}\n\nAnswer:"
        )

        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}\nsource:{source}",
        )
        combine_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=prompt_template,
            document_variable_name="context",
            document_prompt=document_prompt,
        )

        return create_retrieval_chain(
            combine_docs_chain=combine_chain, retriever=retriever
        )
