"""
LangChain utility functions for the ragger application.
Provides low-level functions for vector store loading,
retrieval chain creation, and LLM configuration.
"""

from pathlib import Path

from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from ragger.utils.logging_config import get_logger
from ragger.core.exceptions import EmbeddingModelError, VectorStoreLoadError, VectorStoreNotFoundError
from ragger.config import (DEFAULT_NUM_CHUNKS, LLM_TIMEOUT, OLLAMA_BASE_URL, STOP_SEQUENCE, SYSTEM_PROMPT_CONTEXT,
                           SYSTEM_PROMPT_HISTORY, SYSTEM_PROMPT_INTRO, SYSTEM_PROMPT_QUERY)


def load_vectorstore(db_path: str, embedding_model_name: str) -> FAISS:
    """Load FAISS vector store with the specified embedding model.

    Args:
        db_path: Path to the vector database directory
        embedding_model_name: Name of the HuggingFace embedding model

    Returns:
        FAISS vector store instance
        
    Raises:
        VectorStoreNotFoundError: If the vector store files don't exist
        EmbeddingModelError: If the embedding model fails to load
        VectorStoreLoadError: If the vector store fails to load
    """
    logger = get_logger(__name__)
    
    # Validate that the vector store path exists
    db_path_obj = Path(db_path)
    if not db_path_obj.exists():
        raise VectorStoreNotFoundError(
            f"Vector store directory does not exist: {db_path}",
            path=db_path,
            embedding_model=embedding_model_name
        )
    
    # Check for essential FAISS files
    index_file = db_path_obj / "index.faiss"
    pkl_file = db_path_obj / "index.pkl"
    
    if not index_file.exists():
        raise VectorStoreNotFoundError(
            f"FAISS index file not found: {index_file}",
            path=db_path,
            embedding_model=embedding_model_name
        )
    
    if not pkl_file.exists():
        raise VectorStoreNotFoundError(
            f"FAISS pickle file not found: {pkl_file}",
            path=db_path,
            embedding_model=embedding_model_name
        )
    
    # Initialize embedding model
    try:
        logger.info(f"Loading embedding model: {embedding_model_name}")
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            encode_kwargs={"normalize_embeddings": True}
        )
    except Exception as e:
        raise EmbeddingModelError(
            f"Failed to load embedding model: {str(e)}",
            model_name=embedding_model_name
        ) from e
    
    # Load the vector store
    try:
        logger.info(f"Loading FAISS vector store from: {db_path}")
        vectorstore = FAISS.load_local(
            folder_path=db_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Successfully loaded vector store with {vectorstore.index.ntotal} vectors")
        return vectorstore
    except Exception as e:
        raise VectorStoreLoadError(
            f"Failed to load vector store: {str(e)}",
            path=db_path,
            embedding_model=embedding_model_name
        ) from e


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

    # Initialize the LLM
    try:
        llm = OllamaLLM(
            model=llm_model_name,
            stop=STOP_SEQUENCE,
            timeout=LLM_TIMEOUT,
            base_url=OLLAMA_BASE_URL
        )

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error initializing LLM model: {str(e)}")
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


def create_qa_chain(llm_model_name: str, conversation_history: list = None) -> object:
    """Create a QA chain without retrieval for pre-selected chunks.

    Args:
        llm_model_name: Name of the Ollama LLM model to use
        conversation_history: Optional list of previous Q&A pairs

    Returns:
        QA chain for answering questions with provided context
    """
    # Build the system prompt from components
    template = SYSTEM_PROMPT_INTRO

    # Add history section if available
    if conversation_history and len(conversation_history) > 0:
        template += SYSTEM_PROMPT_HISTORY

    # Add context and query sections
    template += SYSTEM_PROMPT_CONTEXT + SYSTEM_PROMPT_QUERY

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

    # Initialize the LLM
    try:
        llm = OllamaLLM(
            model=llm_model_name,
            stop=STOP_SEQUENCE,
            timeout=LLM_TIMEOUT,
            base_url=OLLAMA_BASE_URL
        )
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error initializing LLM model: {str(e)}")
        raise e

    # Create the question-answer chain
    qa_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt_template,
    )

    return qa_chain