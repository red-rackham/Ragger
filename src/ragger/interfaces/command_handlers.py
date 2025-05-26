"""
Command handlers for the Ragger CLI interface.

This module implements the Command Handler pattern to separate command processing
logic from the main CLI interface, improving maintainability and testability.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ragger.core.commands import Command
from ragger.utils.logging_config import get_logger


class CommandResult:
    """Result of command execution with success status and optional data."""
    
    def __init__(self, success: bool = True, message: str = "", data: Dict[str, Any] = None):
        self.success = success
        self.message = message
        self.data = data or {}
        
    def __str__(self):
        status = "SUCCESS" if self.success else "FAILURE"
        return f"CommandResult({status}: {self.message})"


class CommandHandler(ABC):
    """Abstract base class for all command handlers."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
    
    @abstractmethod
    def get_supported_command_types(self) -> List[str]:
        """Return list of command class names this handler supports."""
        pass
    
    def validate(self, command: Command) -> bool:
        """Validate command before handling. Override for custom validation."""
        return True
    
    def handle(self, command: Command, cli_interface) -> CommandResult:
        """Handle the command and return result."""
        try:
            # Validate command first
            if not self.validate(command):
                return CommandResult(False, "Invalid command parameters")
            
            # Log command handling
            self.logger.debug(f"Handling command: {command.__class__.__name__}")
            
            # Execute the command
            result = self._execute(command, cli_interface)
            
            # Log result
            if result.success:
                self.logger.debug(f"Command completed successfully: {result.message}")
            else:
                self.logger.warning(f"Command failed: {result.message}")
            
            return result
            
        except Exception as e:
            error_msg = f"Handler {self.__class__.__name__} failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return CommandResult(False, error_msg)
    
    @abstractmethod
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute the validated command. Override this method."""
        pass


class CommandHandlerRegistry:
    """Registry to map command types to their handlers."""
    
    def __init__(self):
        self._handlers = {}
        self.logger = get_logger(__name__)
        self._register_default_handlers()
    
    def register(self, handler: CommandHandler):
        """Register a handler for multiple command types."""
        supported_types = handler.get_supported_command_types()
        for command_type in supported_types:
            if command_type in self._handlers:
                self.logger.warning(f"Overriding handler for command type: {command_type}")
            self._handlers[command_type] = handler
            self.logger.debug(f"Registered {handler.__class__.__name__} for {command_type}")
    
    def get_handler(self, command_type: str) -> CommandHandler:
        """Get handler for command type."""
        handler = self._handlers.get(command_type)
        if not handler:
            self.logger.warning(f"No handler found for command type: {command_type}")
        return handler
    
    def get_registered_types(self) -> List[str]:
        """Get all registered command types."""
        return list(self._handlers.keys())
    
    def _register_default_handlers(self):
        """Register all default command handlers."""
        # Import handlers here to avoid circular imports
        from ragger.interfaces.handlers.chunk_management import (
            AddCommandHandler, ChunkDisplayHandler)
        from ragger.interfaces.handlers.configuration import \
            ConfigurationHandler
        from ragger.interfaces.handlers.file_operations import \
            FileOperationsHandler
        from ragger.interfaces.handlers.history import (ClearCommandHandler,
                                                        HistoryCommandHandler)
        from ragger.interfaces.handlers.prompt import (AddPromptCommandHandler,
                                                       PromptCommandHandler)
        from ragger.interfaces.handlers.search import SearchCommandHandler

        # Register all handlers
        self.register(ConfigurationHandler())
        self.register(FileOperationsHandler())
        self.register(HistoryCommandHandler())
        self.register(ClearCommandHandler())
        self.register(PromptCommandHandler())
        self.register(AddPromptCommandHandler())
        self.register(ChunkDisplayHandler())
        self.register(AddCommandHandler())
        self.register(SearchCommandHandler())
        
        self.logger.info(f"Registered {len(self._handlers)} command handlers for {len(self.get_registered_types())} command types")