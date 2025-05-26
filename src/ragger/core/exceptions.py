"""
Custom exceptions for the Ragger application.

This module defines specific exception types for better error handling
and user feedback throughout the application.
"""


class RaggerError(Exception):
    """Base exception class for all Ragger-specific errors."""
    pass


class VectorStoreError(RaggerError):
    """Exception raised when vector store operations fail."""
    
    def __init__(self, message: str, path: str = None, embedding_model: str = None):
        super().__init__(message)
        self.path = path
        self.embedding_model = embedding_model
        
    def __str__(self):
        base_msg = super().__str__()
        if self.path:
            base_msg += f" (Path: {self.path})"
        if self.embedding_model:
            base_msg += f" (Embedding model: {self.embedding_model})"
        return base_msg


class VectorStoreNotFoundError(VectorStoreError):
    """Exception raised when vector store files are not found."""
    pass


class VectorStoreLoadError(VectorStoreError):
    """Exception raised when vector store loading fails."""
    pass


class EmbeddingModelError(RaggerError):
    """Exception raised when embedding model operations fail."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message)
        self.model_name = model_name
        
    def __str__(self):
        base_msg = super().__str__()
        if self.model_name:
            base_msg += f" (Model: {self.model_name})"
        return base_msg


class LLMError(RaggerError):
    """Exception raised when LLM operations fail."""
    
    def __init__(self, message: str, model_name: str = None):
        super().__init__(message)
        self.model_name = model_name
        
    def __str__(self):
        base_msg = super().__str__()
        if self.model_name:
            base_msg += f" (Model: {self.model_name})"
        return base_msg


class ConfigurationError(RaggerError):
    """Exception raised when configuration is invalid."""
    pass