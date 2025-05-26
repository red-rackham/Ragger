"""
Core RAG (Retrieval-Augmented Generation) functionality for the Ragger application.

This module contains the core RAG service that can be used by different interfaces.
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ragger.config import CHUNK_PREVIEW_LENGTH_SHORT, DEFAULT_CONTEXT_SIZE
from ragger.core.commands import ContentProcessor
from ragger.core.exceptions import (EmbeddingModelError, VectorStoreError,
                                    VectorStoreLoadError,
                                    VectorStoreNotFoundError)
from ragger.utils.logging_config import get_logger


class RagService:
    """Core RAG functionality as a service."""
    
    def __init__(self, db_path: str, embedding_model: str = None, llm_model: str = None, num_chunks: int = 5, 
                 ignore_custom: bool = False, combine_chunks: bool = False):
        """Initialize the RAG service.
        
        Args:
            db_path: Path to the vector database
            embedding_model: Name of the embedding model to use (auto-detected if None)
            llm_model: Name of the LLM model to use (None for search-only mode)
            num_chunks: Number of chunks to retrieve for each query
            ignore_custom: Always search vector database, ignore custom chunks
            combine_chunks: Search vector database AND include custom chunks
        """
        self.db_path = db_path
        self.llm_model = llm_model
        self.num_chunks = num_chunks
        self.ignore_custom = ignore_custom
        self.combine_chunks = combine_chunks
        
        # Auto-detect embedding model if not provided
        self.db_info = self.load_ragger_info()
        auto_detected = self.get_auto_detected_embedding_model()
        
        # Create logger early for debugging
        self.logger = get_logger(__name__)
        
        # Determine embedding model: explicit > auto-detected > default fallback
        if embedding_model:
            self.embedding_model = embedding_model
            self.logger.info(f"Using explicitly specified embedding model: {embedding_model}")
        elif auto_detected:
            self.embedding_model = auto_detected
            self.logger.info(f"Auto-detected embedding model from ragger.info: {auto_detected}")
        else:
            # Import here to avoid circular imports
            from ragger.config import DEFAULT_EMBEDDING_MODEL
            self.embedding_model = DEFAULT_EMBEDDING_MODEL
            self.logger.warning(f"No embedding model provided and auto-detection failed, using default: {DEFAULT_EMBEDDING_MODEL}")
            self.logger.warning(f"ragger.info content: {self.db_info}")
            self.logger.warning(f"ragger.info path: {Path(self.db_path) / 'ragger.info'}")
        
        if not self.embedding_model:
            raise ValueError(f"No embedding model could be determined. Available info: {self.db_info}")
        
        # State
        self.vectorstore = None
        self.conversation_history = []
        self.retrieved_chunks_history = []
        self.last_expanded_context = None
        self.last_expanded_chunk_num = None
        
        # Custom chunk list - separate from retrieved chunks
        self.custom_chunks = []  # List to store user-selected chunks
        self.last_custom_chunk_context = None  # Last expanded context from custom chunks
        self.last_custom_chunk_num = None  # Last expanded custom chunk number
        
        # Utilities  
        self.content_processor = ContentProcessor()
        
        # Load the vector store
        self.load_vectorstore()
    
    def load_ragger_info(self) -> Dict[str, Any]:
        """Load ragger.info metadata from the vector database directory.
        
        Returns:
            Dictionary with database metadata or empty dict if not found
        """
        try:
            info_path = Path(self.db_path) / "ragger.info"
            if info_path.exists():
                with open(info_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load ragger.info: {e}")
        return {}
    
    def get_auto_detected_embedding_model(self) -> Optional[str]:
        """Get the embedding model from ragger.info metadata.
        
        Returns:
            Embedding model name or None if not found
        """
        return self.db_info.get('embedding_model')
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the loaded database.
        
        Returns:
            Dictionary with database information
        """
        info = {
            'db_path': self.db_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'num_chunks': self.num_chunks
        }
        
        # Add metadata from ragger.info if available
        if self.db_info:
            info['metadata'] = self.db_info
            
        return info
    
    def load_vectorstore(self):
        """Load the vector store from the specified path.
        
        Raises:
            VectorStoreNotFoundError: If vector store files don't exist
            EmbeddingModelError: If embedding model fails to load
            VectorStoreLoadError: If vector store loading fails
        """
        from ragger.utils.langchain_utils import load_vectorstore
        
        self.logger.info(f"Loading vector store from: {self.db_path}")
        self.logger.info(f"Using embedding model: {self.embedding_model}")
        
        # Let specific exceptions bubble up - no silent failures
        self.vectorstore = load_vectorstore(self.db_path, self.embedding_model)
        self.logger.info("Vector store loaded successfully")
    
    def create_rag_chain(self, conversation_history=None):
        """Create a RAG chain with the specified parameters.
        
        Args:
            conversation_history: Optional conversation history to use
            
        Returns:
            Tuple of (rag_chain, retriever)
        """
        from ragger.utils.langchain_utils import create_rag_chain
        
        if not self.llm_model:
            raise ValueError("No LLM model specified. Cannot create RAG chain.")
        
        history = conversation_history or self.conversation_history
        return create_rag_chain(
            self.vectorstore, 
            self.llm_model,
            self.num_chunks,
            history
        )
    
    def _create_qa_chain(self, conversation_history=None):
        """Create a QA chain without retrieval for pre-selected chunks."""
        from ragger.utils.langchain_utils import create_qa_chain
        
        if not self.llm_model:
            raise ValueError("No LLM model specified. Cannot create QA chain.")
        
        history = conversation_history or self.conversation_history
        return create_qa_chain(self.llm_model, history)
    
    def _determine_chunks_for_query(self, query: str) -> Dict[str, Any]:
        """Smart chunk selection based on available custom chunks and user preferences.
        
        Args:
            query: The search query
            
        Returns:
            Dictionary with chunk selection results including:
            - use_custom: Whether to use custom chunks
            - use_search: Whether to perform search
            - chunks: List of chunks to use
            - search_results: Retrieved chunks from search (if any)
            - custom_chunks: Custom chunks (if any)
            - mode: Description of the mode used
        """
        result = {
            'use_custom': False,
            'use_search': False,
            'chunks': [],
            'search_results': [],
            'custom_chunks': [],
            'mode': '',
            'success': True,
            'error': None
        }
        
        try:
            if self.combine_chunks:
                # Always search AND include custom chunks
                result['use_search'] = True
                result['use_custom'] = bool(self.custom_chunks)
                result['mode'] = 'combine'
                
                # Perform search
                search_result = self._perform_search(query)
                if search_result['success']:
                    result['search_results'] = search_result['retrieved_docs']
                else:
                    result['success'] = False
                    result['error'] = search_result['error']
                    return result
                
                # Combine search results with custom chunks
                result['custom_chunks'] = self.custom_chunks
                result['chunks'] = self._combine_chunks(self.custom_chunks, result['search_results'])
                
            elif self.ignore_custom:
                # Always search, ignore custom chunks
                result['use_search'] = True
                result['mode'] = 'search_only'
                
                search_result = self._perform_search(query)
                if search_result['success']:
                    result['search_results'] = search_result['retrieved_docs']
                    result['chunks'] = result['search_results']
                else:
                    result['success'] = False
                    result['error'] = search_result['error']
                    return result
                    
            else:
                # Default smart behavior: use custom chunks if available, otherwise search
                if self.custom_chunks:
                    # Use custom chunks only
                    result['use_custom'] = True
                    result['mode'] = 'custom_only'
                    result['custom_chunks'] = self.custom_chunks
                    result['chunks'] = self.custom_chunks
                else:
                    # No custom chunks, fall back to search
                    result['use_search'] = True
                    result['mode'] = 'search_fallback'
                    
                    search_result = self._perform_search(query)
                    if search_result['success']:
                        result['search_results'] = search_result['retrieved_docs']
                        result['chunks'] = result['search_results']
                    else:
                        result['success'] = False
                        result['error'] = search_result['error']
                        return result
                        
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            
        return result
    
    def _perform_search(self, query: str) -> Dict[str, Any]:
        """Perform vector database search.
        
        Args:
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        # Clean any command text from the query
        clean_text = self.content_processor.clean_commands(query)
        
        start_time = time.time()
        
        try:
            # Create retriever
            if not self.llm_model:
                # For search-only mode, create retriever directly
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.num_chunks})
            else:
                # Create the RAG chain to get the retriever
                rag_chain, retriever = self.create_rag_chain()
            
            # Retrieve documents
            retrieved_docs = retriever.invoke(clean_text)
            retrieval_time = time.time() - start_time
            
            return {
                'success': True,
                'retrieved_docs': retrieved_docs,
                'retrieval_time': retrieval_time,
                'query': clean_text
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'retrieved_docs': [],
                'retrieval_time': time.time() - start_time,
                'query': clean_text
            }
    
    def _combine_chunks(self, custom_chunks: list, retrieved_chunks: list) -> list:
        """Combine custom and retrieved chunks, removing duplicates.
        
        Args:
            custom_chunks: List of custom chunks
            retrieved_chunks: List of retrieved chunks
            
        Returns:
            Combined list with custom chunks first, then unique retrieved chunks
        """
        if not custom_chunks:
            return retrieved_chunks
        if not retrieved_chunks:
            return custom_chunks
            
        # Start with custom chunks
        combined = list(custom_chunks)
        
        # Add retrieved chunks that don't duplicate custom chunks
        for retrieved_chunk in retrieved_chunks:
            is_duplicate = False
            for custom_chunk in custom_chunks:
                # Check for content similarity (simple approach)
                if (retrieved_chunk.page_content.strip() == custom_chunk.page_content.strip() or
                    retrieved_chunk.metadata.get('source') == custom_chunk.metadata.get('source') and
                    retrieved_chunk.metadata.get('chunk_index') == custom_chunk.metadata.get('chunk_index')):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                combined.append(retrieved_chunk)
                
        return combined
    
    def search_only(self, text: str) -> Dict[str, Any]:
        """Retrieve relevant chunks without LLM generation using smart chunk selection.
        
        Args:
            text: Query text
            
        Returns:
            Dictionary with retrieval results including mode information
        """
        start_time = time.time()
        
        # Use smart chunk selection
        chunk_result = self._determine_chunks_for_query(text)
        total_time = time.time() - start_time
        
        if not chunk_result['success']:
            return {
                'success': False,
                'error': chunk_result['error'],
                'retrieved_docs': [],
                'retrieval_time': total_time,
                'query': self.content_processor.clean_commands(text),
                'mode': 'error'
            }
        
        return {
            'success': True,
            'retrieved_docs': chunk_result['chunks'],
            'search_results': chunk_result['search_results'],
            'custom_chunks': chunk_result['custom_chunks'],
            'retrieval_time': total_time,
            'query': self.content_processor.clean_commands(text),
            'mode': chunk_result['mode'],
            'use_custom': chunk_result['use_custom'],
            'use_search': chunk_result['use_search']
        }
    
    def generate_response(self, query: str, use_history: bool = True) -> Dict[str, Any]:
        """Generate LLM response using smart chunk selection.
        
        Args:
            query: Original query text
            use_history: Whether to use conversation history
            
        Returns:
            Dictionary with generation results including chunk selection info
        """
        if not self.llm_model:
            return {
                'success': False,
                'error': 'No LLM model loaded. Use search commands instead.',
                'answer': 'No LLM model loaded. Use search commands instead.',
                'no_llm_model': True,
                'generation_time': 0
            }
        
        # Use smart chunk selection to get chunks
        chunk_result = self._determine_chunks_for_query(query)
        if not chunk_result['success']:
            return {
                'success': False,
                'error': chunk_result['error'],
                'answer': f"Error selecting chunks: {chunk_result['error']}",
                'generation_time': 0,
                'mode': 'error'
            }
        
        retrieved_docs = chunk_result['chunks']
        
        # Clean any command text from the query
        clean_text = self.content_processor.clean_commands(query)
        
        # Create QA chain without retrieval
        try:
            qa_chain = self._create_qa_chain(
                self.conversation_history if use_history else None
            )
        except ValueError as e:
            return {
                'success': False,
                'error': str(e),
                'answer': str(e),
                'no_llm_model': True,
                'generation_time': 0
            }
        
        # Process query with pre-selected chunks
        try:
            chain_input = {
                "input": clean_text,
                "context": retrieved_docs
            }
            if use_history and self.conversation_history:
                history_text = "\n\n".join([f"User Prompt: {q}\nAssistant Response: {a}" 
                                         for q, a in self.conversation_history])
                chain_input["history"] = history_text
            
            start_time = time.time()
            answer = qa_chain.invoke(chain_input)
            elapsed_time = time.time() - start_time
            
            answer = answer.strip() if isinstance(answer, str) else str(answer).strip()
            
            # Update history if requested
            if use_history:
                self.conversation_history.append((clean_text, answer))
                self.retrieved_chunks_history.append(list(retrieved_docs) if retrieved_docs else [])
            
            return {
                'success': True,
                'answer': answer,
                'generation_time': elapsed_time,
                'mode': chunk_result['mode'],
                'use_custom': chunk_result['use_custom'],
                'use_search': chunk_result['use_search'],
                'custom_chunks': chunk_result['custom_chunks'],
                'search_results': chunk_result['search_results'],
                'retrieved_docs': retrieved_docs
            }
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            return {
                'success': False,
                'error': str(e),
                'answer': error_msg,
                'generation_time': 0,
                'mode': chunk_result.get('mode', 'error')
            }
    
    def query(self, text: str, use_history: bool = True) -> Dict[str, Any]:
        """Process a query and return relevant chunks and response.
        
        Args:
            text: Query text
            use_history: Whether to use conversation history
            
        Returns:
            Dictionary with query results
        """
        if not self.llm_model:
            return {
                'success': False,
                'error': 'No LLM model loaded. Use search commands instead.',
                'retrieved_docs': [],
                'answer': 'No LLM model loaded. Use search commands instead.',
                'no_llm_model': True
            }
        
        # Clean any command text from the query
        clean_text = self.content_processor.clean_commands(text)
        
        # Create the RAG chain
        try:
            rag_chain, retriever = self.create_rag_chain(
                self.conversation_history if use_history else None
            )
        except ValueError as e:
            return {
                'success': False,
                'error': str(e),
                'retrieved_docs': [],
                'answer': str(e),
                'no_llm_model': True
            }
        
        start_time = time.time()
        
        # Retrieve documents
        try:
            retrieved_docs = retriever.invoke(clean_text)
            retrieval_time = time.time() - start_time
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'retrieved_docs': [],
                'answer': f"Error retrieving documents: {str(e)}"
            }
        
        # Process the query with the LLM
        try:
            # Prepare input for the chain
            chain_input = {"input": clean_text}
            if use_history and self.conversation_history:
                history_text = "\n\n".join([f"User Prompt: {q}\nAssistant Response: {a}" 
                                         for q, a in self.conversation_history])
                chain_input["history"] = history_text
            
            # Generate response
            start_time = time.time()
            result = rag_chain.invoke(chain_input)
            elapsed_time = time.time() - start_time
            
            answer = result.get("answer", "").strip()
            
            # Update history if requested
            if use_history:
                self.conversation_history.append((clean_text, answer))
                self.retrieved_chunks_history.append(list(retrieved_docs) if retrieved_docs else [])
            
            return {
                'success': True,
                'retrieved_docs': retrieved_docs,
                'answer': answer,
                'retrieval_time': retrieval_time,
                'generation_time': elapsed_time
            }
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            return {
                'success': False,
                'error': str(e),
                'retrieved_docs': retrieved_docs,
                'answer': error_msg
            }
    
    def _validate_chunk_number(self, chunk_num: int, chunks: list, chunk_type: str) -> Optional[Dict[str, Any]]:
        """Validate chunk number and return error dict if invalid.
        
        Args:
            chunk_num: The chunk number to validate
            chunks: List of chunks to validate against
            chunk_type: Type description for error messages
            
        Returns:
            Error dict if invalid, None if valid
        """
        if chunk_num < 1 or chunk_num > len(chunks):
            return {
                'success': False,
                'error': f"Invalid chunk number. Available {chunk_type}: 1-{len(chunks)}",
                'expanded_context': None,
                'chunk_num': chunk_num
            }
        return None
    
    def _expand_chunk_with_source(self, chunk, chunk_num: int, context_size: int) -> Dict[str, Any]:
        """Expand chunk with source path using terminal utilities.
        
        Args:
            chunk: The chunk document to expand
            chunk_num: Chunk number for tracking
            context_size: Size of context window
            
        Returns:
            Dictionary with expansion results
        """
        source_path = chunk.metadata.get("source", None)
        chunk_index = chunk.metadata.get("chunk_index", chunk_num)
        
        if not source_path:
            # No source path - return cleaned content directly
            cleaned_context = self.content_processor.clean_commands(chunk.page_content)
            return {
                'success': True,
                'expanded_context': cleaned_context,
                'chunk_num': chunk_num,
                'source_path': None,
                'chunk_index': chunk_index,
                'cleaned_context': cleaned_context
            }
        
        # Expand using source path
        from ragger.ui.terminal import expand_chunk_context, get_terminal_width
        try:
            terminal_width = get_terminal_width()
            expanded_context = expand_chunk_context(
                chunk, source_path, chunk_index, terminal_width, context_size
            )
            
            cleaned_context = self.content_processor.clean_commands(expanded_context)
            return {
                'success': True,
                'expanded_context': cleaned_context,
                'chunk_num': chunk_num,
                'source_path': source_path,
                'chunk_index': chunk_index,
                'cleaned_context': cleaned_context
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'expanded_context': None,
                'chunk_num': chunk_num
            }
    
    def _update_expansion_state(self, cleaned_context: str, chunk_num: int, is_custom: bool) -> None:
        """Update the last expanded chunk tracking state.
        
        Args:
            cleaned_context: The cleaned expanded context
            chunk_num: Chunk number being tracked
            is_custom: Whether this is a custom chunk
        """
        if is_custom:
            self.last_custom_chunk_context = cleaned_context
            self.last_custom_chunk_num = chunk_num
        else:
            self.last_expanded_context = cleaned_context
            self.last_expanded_chunk_num = chunk_num

    def expand_chunk(self, chunk_num: int, context_size: int = DEFAULT_CONTEXT_SIZE, from_custom: bool = False) -> Dict[str, Any]:
        """Expand a chunk from the most recent query results.
        
        Args:
            chunk_num: Number of the chunk to expand
            context_size: Size of the context window
            from_custom: Whether to expand from the custom chunks list (kept for backward compatibility)
            
        Returns:
            Dictionary with expansion results
        """
        # For backward compatibility, redirect to custom chunks if requested
        if from_custom:
            return self.expand_custom_chunk(chunk_num, context_size)
            
        # Check if we have any retrieved chunks
        if not self.retrieved_chunks_history:
            return {
                'success': False,
                'error': "No retrieved chunks available",
                'expanded_context': None,
                'chunk_num': chunk_num
            }
        
        # Find the most recent query with retrieved chunks
        recent_chunks = None
        for chunks in reversed(self.retrieved_chunks_history):
            if chunks:
                recent_chunks = chunks
                break
        
        if not recent_chunks:
            return {
                'success': False,
                'error': "No chunks available from previous queries",
                'expanded_context': None,
                'chunk_num': chunk_num
            }
        
        # Validate chunk number
        validation_error = self._validate_chunk_number(chunk_num, recent_chunks, "chunks")
        if validation_error:
            return validation_error
        
        # Get the requested chunk
        chunk = recent_chunks[chunk_num - 1]  # Convert to 0-based index
        
        # Expand chunk with source
        result = self._expand_chunk_with_source(chunk, chunk_num, context_size)
        if not result['success']:
            return result
        
        # Update state tracking
        self._update_expansion_state(result['cleaned_context'], chunk_num, is_custom=False)
        
        # Add from_custom flag and return
        result['from_custom'] = False
        return result
    
    def get_last_expanded_context(self) -> Optional[Dict[str, Any]]:
        """Get the last expanded context if available.
        
        Returns:
            Dictionary with last expanded context or None
        """
        if self.last_expanded_context is None or self.last_expanded_chunk_num is None:
            return None
            
        return {
            'expanded_context': self.last_expanded_context,
            'chunk_num': self.last_expanded_chunk_num
        }
    
    def add_context_to_prompt(self, context: str, user_input: str, position: str = "start") -> str:
        """Add context to a user prompt.
        
        Args:
            context: Context to add (expanded chunk)
            user_input: User input text
            position: Where to add context ("start" or "end")
            
        Returns:
            Combined prompt text
        """
        # Clean context of any command text
        clean_context = self.content_processor.clean_commands(context)
        
        # Format as a default prompt
        default_text = f"Regarding this content:\n\n{clean_context}\n\n"
        
        if position == "end":
            # Append context at the end
            if user_input.strip():
                return f"{user_input}\n\n{default_text}"
            else:
                return default_text
        else:
            # Default: prepend context at the start
            return f"{default_text}{user_input}"
    
    def add_to_custom_chunks(self, chunk, source_path: str = None) -> Dict[str, Any]:
        """Add a chunk to the custom chunks list.
        
        Args:
            chunk: The chunk to add (Document object)
            source_path: Optional source path for the chunk
            
        Returns:
            Dictionary with operation results
        """
        try:
            if chunk is None:
                return {
                    'success': False,
                    'error': "No chunk provided",
                    'chunks_count': len(self.custom_chunks)
                }
                
            # Add the chunk to our custom list
            self.custom_chunks.append(chunk)
            
            return {
                'success': True,
                'chunks_count': len(self.custom_chunks),
                'chunk_num': len(self.custom_chunks),  # 1-based index of the added chunk
                'chunk_preview': chunk.page_content[:CHUNK_PREVIEW_LENGTH_SHORT] + "..." if len(chunk.page_content) > CHUNK_PREVIEW_LENGTH_SHORT else chunk.page_content
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'chunks_count': len(self.custom_chunks)
            }
    
    def remove_from_custom_chunks(self, chunk_num: int) -> Dict[str, Any]:
        """Remove a chunk from the custom chunks list.
        
        Args:
            chunk_num: The 1-based index of the chunk to remove
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Convert to 0-based index
            idx = chunk_num - 1
            
            # Check if the index is valid
            if idx < 0 or idx >= len(self.custom_chunks):
                return {
                    'success': False,
                    'error': f"Invalid chunk number. Available chunks: 1-{len(self.custom_chunks)}",
                    'chunks_count': len(self.custom_chunks)
                }
            
            # Remove the chunk
            removed_chunk = self.custom_chunks.pop(idx)
            
            return {
                'success': True,
                'chunks_count': len(self.custom_chunks),
                'removed_chunk_preview': removed_chunk.page_content[:CHUNK_PREVIEW_LENGTH_SHORT] + "..." if len(removed_chunk.page_content) > CHUNK_PREVIEW_LENGTH_SHORT else removed_chunk.page_content
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'chunks_count': len(self.custom_chunks)
            }
    
    def get_custom_chunks(self) -> Dict[str, Any]:
        """Get the list of custom chunks.
        
        Returns:
            Dictionary with custom chunks information
        """
        return {
            'success': True,
            'chunks': self.custom_chunks,
            'chunks_count': len(self.custom_chunks)
        }
    
    def clear_custom_chunks(self) -> Dict[str, Any]:
        """Clear the custom chunks list.
        
        Returns:
            Dictionary with operation results
        """
        previous_count = len(self.custom_chunks)
        self.custom_chunks = []
        self.last_custom_chunk_context = None
        self.last_custom_chunk_num = None
        
        return {
            'success': True,
            'previous_count': previous_count,
            'current_count': 0
        }
        
    def expand_custom_chunk(self, chunk_num: int, context_size: int = DEFAULT_CONTEXT_SIZE) -> Dict[str, Any]:
        """Expand a chunk from the custom chunks list.
        
        Args:
            chunk_num: Number of the chunk to expand (1-based)
            context_size: Size of the context window
            
        Returns:
            Dictionary with expansion results
        """
        # Check if we have any custom chunks
        if not self.custom_chunks:
            return {
                'success': False,
                'error': "No custom chunks available",
                'expanded_context': None,
                'chunk_num': chunk_num
            }
        
        # Validate chunk number
        validation_error = self._validate_chunk_number(chunk_num, self.custom_chunks, "custom chunks")
        if validation_error:
            return validation_error
        
        # Get the requested chunk (convert to 0-based index)
        chunk = self.custom_chunks[chunk_num - 1]
        
        # Expand chunk with source
        result = self._expand_chunk_with_source(chunk, chunk_num, context_size)
        if not result['success']:
            return result
        
        # Update state tracking
        self._update_expansion_state(result['cleaned_context'], chunk_num, is_custom=True)
        
        return result
    
    def get_last_expanded_custom_chunk(self) -> Optional[Dict[str, Any]]:
        """Get the last expanded custom chunk context if available.
        
        Returns:
            Dictionary with last expanded context or None
        """
        if self.last_custom_chunk_context is None or self.last_custom_chunk_num is None:
            return None
            
        return {
            'expanded_context': self.last_custom_chunk_context,
            'chunk_num': self.last_custom_chunk_num
        }
    
    def clear_history(self, preserve_custom_chunks: bool = False) -> Dict[str, Any]:
        """Clear conversation and retrieval history.
        
        Args:
            preserve_custom_chunks: Whether to preserve custom chunks
            
        Returns:
            Dictionary with operation results
        """
        self.conversation_history = []
        self.retrieved_chunks_history = []
        self.last_expanded_context = None
        self.last_expanded_chunk_num = None
        
        custom_chunks_count = len(self.custom_chunks)
        if not preserve_custom_chunks:
            self.custom_chunks = []
            self.last_custom_chunk_context = None
            self.last_custom_chunk_num = None
        
        return {
            'success': True,
            'preserved_custom_chunks': preserve_custom_chunks,
            'custom_chunks_count': custom_chunks_count if preserve_custom_chunks else 0
        }
    
    def save_history(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """Save conversation history to a file.
        
        Args:
            filepath: Optional file path to save to
            
        Returns:
            Dictionary with save results
        """
        from ragger.utils.conversation_history import save_conversation_history
        
        if not self.conversation_history:
            return {
                'success': False,
                'error': "No conversation history to save",
                'filepath': None
            }
        
        try:
            saved_file = save_conversation_history(
                self.conversation_history,
                self.retrieved_chunks_history,
                self.llm_model,
                filepath,
                self.custom_chunks  # Pass custom chunks for saving
            )
            return {
                'success': True,
                'filepath': saved_file,
                'custom_chunks_count': len(self.custom_chunks)
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filepath': None
            }
            
    def set_chunks(self, num_chunks: int) -> Dict[str, Any]:
        """Set the number of chunks to retrieve.
        
        Args:
            num_chunks: Number of chunks to retrieve
            
        Returns:
            Dictionary with operation results
        """
        if num_chunks < 1:
            return {
                'success': False,
                'error': "Number of chunks must be at least 1",
                'previous_value': self.num_chunks,
                'current_value': self.num_chunks
            }
            
        # Store the previous value for the result
        previous_value = self.num_chunks
        
        # Update the number of chunks
        self.num_chunks = num_chunks
        
        return {
            'success': True,
            'previous_value': previous_value,
            'current_value': self.num_chunks,
            'message': f"Number of chunks to retrieve set to {num_chunks}"
        }