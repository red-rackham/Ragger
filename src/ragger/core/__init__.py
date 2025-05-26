"""
Core functionality for the Ragger application.

This package contains the core RAG functionality, command processing,
and exception handling.
"""
from ragger.core.commands import Command, CommandManager
from ragger.core.completer import CommandCompleter
from ragger.core.exceptions import (ConfigurationError, EmbeddingModelError,
                                    LLMError, RaggerError, VectorStoreError,
                                    VectorStoreLoadError,
                                    VectorStoreNotFoundError)
from ragger.core.rag import RagService

__all__ = [
    'RagService',
    'CommandManager', 
    'Command',
    'CommandCompleter',
    'RaggerError',
    'VectorStoreError',
    'VectorStoreNotFoundError',
    'VectorStoreLoadError',
    'EmbeddingModelError',
    'LLMError',
    'ConfigurationError'
]