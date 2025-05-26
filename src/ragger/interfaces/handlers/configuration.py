"""
Configuration command handlers for chunk settings and app configuration.
"""

from typing import List

from ragger.core.commands import Command, GetChunksCommand, SetChunksCommand
from ragger.interfaces.command_handlers import CommandHandler, CommandResult
from ragger.ui.resources import Emojis


class ConfigurationHandler(CommandHandler):
    """Handles configuration commands (chunks settings)."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['SetChunksCommand', 'GetChunksCommand']
    
    def validate(self, command: Command) -> bool:
        """Validate configuration commands."""
        if isinstance(command, SetChunksCommand):
            return command.num_chunks is not None and command.num_chunks > 0
        return True
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute configuration command."""
        if isinstance(command, SetChunksCommand):
            return self._handle_set_chunks(command, cli_interface)
        elif isinstance(command, GetChunksCommand):
            return self._handle_get_chunks(command, cli_interface)
        else:
            return CommandResult(False, f"Unsupported command type: {command.__class__.__name__}")
    
    def _handle_set_chunks(self, command: SetChunksCommand, cli_interface) -> CommandResult:
        """Handle setting the number of chunks to retrieve."""
        if command.num_chunks is None:
            cli_interface.ui.add_response(
                f"\n{Emojis.ERROR} Please specify the number of chunks to retrieve.", 
                "error"
            )
            return CommandResult(False, "Missing chunk number parameter")
        
        # Call the RAG service to set the number of chunks
        result = cli_interface.rag_service.set_chunks(command.num_chunks)
        
        if result['success']:
            # Show success message with previous and new values
            previous = result['previous_value']
            current = result['current_value']
            cli_interface.ui.add_response(
                f"\n{Emojis.CHECK} Number of chunks to retrieve changed from {previous} to {current}.", 
                "info"
            )
            cli_interface.ui.add_response(
                "Use the '-sc' or '--set-chunks' command-line option to set this value when starting the application.", 
                "info"
            )
            return CommandResult(True, f"Chunks setting updated to {current}")
        else:
            # Show error message
            error_msg = result.get('error', 'Unknown error')
            cli_interface.ui.add_response(
                f"\n{Emojis.ERROR} Error: {error_msg}", 
                "error"
            )
            return CommandResult(False, error_msg)
    
    def _handle_get_chunks(self, command: GetChunksCommand, cli_interface) -> CommandResult:
        """Handle displaying current chunks setting."""
        num_chunks = cli_interface.rag_service.num_chunks
        cli_interface.ui.add_response(
            f"\n{Emojis.CHUNKS} Current number of chunks to retrieve: {num_chunks}", 
            "info"
        )
        cli_interface.ui.add_response(
            "Use 'sc <number>' to change this setting.", 
            "info"
        )
        cli_interface.ui.add_response(
            "Use the '-sc' or '--set-chunks' command-line option to set this value when starting the application.", 
            "info"
        )
        return CommandResult(True, f"Current chunks setting: {num_chunks}")