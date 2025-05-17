"""
Core RAG (Retrieval-Augmented Generation) functionality for the Ragger application.

This module contains the core RAG service that can be used by different interfaces.
"""
import time
from typing import Dict, Any, Optional

# Import from existing code
from ragger.core.commands import ContentProcessor


class RagService:
    """Core RAG functionality as a service."""
    
    def __init__(self, db_path: str, embedding_model: str, llm_model: str, num_chunks: int):
        """Initialize the RAG service.
        
        Args:
            db_path: Path to the vector database
            embedding_model: Name of the embedding model to use
            llm_model: Name of the LLM model to use
            num_chunks: Number of chunks to retrieve for each query
        """
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.num_chunks = num_chunks
        
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
    
    def load_vectorstore(self):
        """Load the vector store from the specified path."""
        from ragger.utils.langchain_utils import load_vectorstore
        
        try:
            self.vectorstore = load_vectorstore(self.db_path, self.embedding_model)
            return True
        except Exception as e:
            # Log error
            print(f"Error loading vector database: {str(e)}")
            return False
    
    def create_rag_chain(self, conversation_history=None):
        """Create a RAG chain with the specified parameters.
        
        Args:
            conversation_history: Optional conversation history to use
            
        Returns:
            Tuple of (rag_chain, retriever)
        """
        from ragger.utils.langchain_utils import create_rag_chain
        
        history = conversation_history or self.conversation_history
        return create_rag_chain(
            self.vectorstore, 
            self.llm_model,
            self.num_chunks,
            history
        )
    
    def query(self, text: str, use_history: bool = True) -> Dict[str, Any]:
        """Process a query and return relevant chunks and response.
        
        Args:
            text: Query text
            use_history: Whether to use conversation history
            
        Returns:
            Dictionary with query results
        """
        # Clean any command text from the query
        clean_text = self.content_processor.clean_commands(text)
        
        # Create the RAG chain
        rag_chain, retriever = self.create_rag_chain(
            self.conversation_history if use_history else None
        )
        
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
    
    def expand_chunk(self, chunk_num: int, context_size: int = 500, from_custom: bool = False) -> Dict[str, Any]:
        """Expand a chunk from the most recent query results or custom chunks.
        
        Args:
            chunk_num: Number of the chunk to expand
            context_size: Size of the context window
            from_custom: Whether to expand from the custom chunks list
            
        Returns:
            Dictionary with expansion results
        """
        # Choose the appropriate source based on from_custom flag
        if from_custom:
            return self.expand_custom_chunk(chunk_num, context_size)
            
        # Expand from retrieved chunks (legacy behavior)
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
        
        # Check if the chunk number is valid
        if chunk_num < 1 or chunk_num > len(recent_chunks):
            return {
                'success': False,
                'error': f"Invalid chunk number. Available chunks: 1-{len(recent_chunks)}",
                'expanded_context': None,
                'chunk_num': chunk_num
            }
        
        # Get the requested chunk
        chunk = recent_chunks[chunk_num - 1]  # Convert to 0-based index
        
        # Get source info
        source_path = chunk.metadata.get("source", None)
        chunk_index = chunk.metadata.get("chunk_index", chunk_num)
        
        if not source_path:
            # If no source path, just return the chunk content
            cleaned_context = self.content_processor.clean_commands(chunk.page_content)
            
            # Update the last expanded chunk tracking
            self.last_expanded_context = cleaned_context
            self.last_expanded_chunk_num = chunk_num
            
            return {
                'success': True,
                'expanded_context': cleaned_context,
                'chunk_num': chunk_num,
                'source_path': None,
                'chunk_index': chunk_index
            }
        
        # Get expanded context
        from ragger.ui.terminal import expand_chunk_context, get_terminal_width
        try:
            terminal_width = get_terminal_width()
            expanded_context = expand_chunk_context(
                chunk, source_path, chunk_index, terminal_width, context_size
            )
            
            # Clean any command text from the expanded context
            cleaned_context = self.content_processor.clean_commands(expanded_context)
            
            # Update the last expanded chunk tracking
            self.last_expanded_context = cleaned_context
            self.last_expanded_chunk_num = chunk_num
            
            return {
                'success': True,
                'expanded_context': cleaned_context,
                'chunk_num': chunk_num,
                'source_path': source_path,
                'chunk_index': chunk_index,
                'from_custom': False
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'expanded_context': None,
                'chunk_num': chunk_num
            }
    
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
                'chunk_preview': chunk.page_content[:50] + "..." if len(chunk.page_content) > 50 else chunk.page_content
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
                'removed_chunk_preview': removed_chunk.page_content[:50] + "..." if len(removed_chunk.page_content) > 50 else removed_chunk.page_content
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
        
    def expand_custom_chunk(self, chunk_num: int, context_size: int = 500) -> Dict[str, Any]:
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
        
        # Check if the chunk number is valid
        if chunk_num < 1 or chunk_num > len(self.custom_chunks):
            return {
                'success': False,
                'error': f"Invalid chunk number. Available custom chunks: 1-{len(self.custom_chunks)}",
                'expanded_context': None,
                'chunk_num': chunk_num
            }
        
        # Get the requested chunk (convert to 0-based index)
        chunk = self.custom_chunks[chunk_num - 1]
        
        # Get source info
        source_path = chunk.metadata.get("source", None)
        chunk_index = chunk.metadata.get("chunk_index", chunk_num)
        
        if not source_path:
            # If no source path, just return the chunk content
            cleaned_context = self.content_processor.clean_commands(chunk.page_content)
            
            # Update the last expanded chunk tracking
            self.last_custom_chunk_context = cleaned_context
            self.last_custom_chunk_num = chunk_num
            
            return {
                'success': True,
                'expanded_context': cleaned_context,
                'chunk_num': chunk_num,
                'source_path': None,
                'chunk_index': chunk_index
            }
            
        # Get expanded context if source path is available
        from ragger.ui.terminal import expand_chunk_context, get_terminal_width
        try:
            terminal_width = get_terminal_width()
            expanded_context = expand_chunk_context(
                chunk, source_path, chunk_index, terminal_width, context_size
            )
            
            # Clean any command text from the expanded context
            cleaned_context = self.content_processor.clean_commands(expanded_context)
            
            # Update the last expanded chunk tracking
            self.last_custom_chunk_context = cleaned_context
            self.last_custom_chunk_num = chunk_num
            
            return {
                'success': True,
                'expanded_context': cleaned_context,
                'chunk_num': chunk_num,
                'source_path': source_path,
                'chunk_index': chunk_index
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'expanded_context': None,
                'chunk_num': chunk_num
            }
    
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