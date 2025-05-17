"""
LangChain utility functions for the ragger application.
Provides low-level functions for vector store loading,
retrieval chain creation, and LLM configuration.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from ragger.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_NUM_CHUNKS,
    STOP_SEQUENCE,
    LLM_TIMEOUT,
    OLLAMA_BASE_URL,
    SYSTEM_PROMPT_INTRO,
    SYSTEM_PROMPT_HISTORY,
    SYSTEM_PROMPT_CONTEXT,
    SYSTEM_PROMPT_QUERY
)
from ragger.ui.resources import Emojis


def load_vectorstore(db_path: str, embedding_model_name: str) -> FAISS:
    """Load FAISS vector store with the specified embedding model.

    Args:
        db_path: Path to the vector database directory
        embedding_model_name: Name of the HuggingFace embedding model

    Returns:
        FAISS vector store instance
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"normalize_embeddings": True}
    )

    return FAISS.load_local(
        folder_path=db_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )


def create_rag_chain(vectorstore: FAISS,
                     llm_model_name: str,
                     num_chunks: int = DEFAULT_NUM_CHUNKS,
                     conversation_history: list = None,
                     system_prompt: str = None) -> tuple[object, object]:
    """Create a RAG chain with conversation history support.

    Args:
        vectorstore: The FAISS vector store for document retrieval
        llm_model_name: Name of the Ollama LLM model to use
        num_chunks: Number of document chunks to retrieve
        conversation_history: Optional list of previous Q&A pairs
        system_prompt: Optional custom prompt to use

    Returns:
        Tuple of (rag_chain, retriever) for question answering
    """
    # Build the system prompt from components if not provided
    if not system_prompt:
        # Start with intro
        template = SYSTEM_PROMPT_INTRO

        # Add history section if available
        if conversation_history and len(conversation_history) > 0:
            template += SYSTEM_PROMPT_HISTORY

        # Add context and query sections
        template += SYSTEM_PROMPT_CONTEXT + SYSTEM_PROMPT_QUERY
    else:
        template = system_prompt

    # Define input variables based on whether we have conversation history
    if conversation_history and len(conversation_history) > 0:
        input_variables = ["history", "context", "input"]
    else:
        input_variables = ["context", "input"]

    # Create the template
    prompt_template = PromptTemplate(
        input_variables=input_variables,
        template=template
    )

    # Initialize the LLM with timeout
    # print(f"\n{Emojis.INFO} Initializing LLM model: {llm_model_name} ({OLLAMA_BASE_URL})...")
    try:
        llm = OllamaLLM(
            model=llm_model_name,
            stop=STOP_SEQUENCE,
            timeout=LLM_TIMEOUT,
            base_url=OLLAMA_BASE_URL
        )

    except Exception as e:
        print(f"\n{Emojis.WARNING} Warning: Error initializing LLM model: {str(e)}")
        raise e

    # Create the question-answer chain
    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template,
    )

    # Configure the retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": num_chunks}
    )

    # Create the RAG chain
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )

    return rag_chain, retriever