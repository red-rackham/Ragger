"""
File operations command handlers for saving conversation history.
"""

from typing import List

from ragger.core.commands import Command, SaveCommand
from ragger.interfaces.command_handlers import CommandHandler, CommandResult


class FileOperationsHandler(CommandHandler):
    """Handles file operations (save)."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['SaveCommand']
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute file operation command."""
        if isinstance(command, SaveCommand):
            return self._handle_save(command, cli_interface)
        else:
            return CommandResult(False, f"Unsupported command type: {command.__class__.__name__}")
    
    def _handle_save(self, command: SaveCommand, cli_interface) -> CommandResult:
        """Handle saving conversation history."""
        if not cli_interface.rag_service.conversation_history:
            cli_interface.ui.add_response("No conversation history to save yet.", "warning")
            return CommandResult(False, "No conversation history available")
        
        # Save the conversation history
        result = cli_interface.rag_service.save_history()
        
        if result['success']:
            cli_interface.ui.add_response(
                f"Conversation history saved to {result['filepath']}", 
                "info"
            )
            return CommandResult(True, f"History saved to {result['filepath']}")
        else:
            error_msg = result.get('error', 'Unknown error')
            cli_interface.ui.add_response(
                f"Error saving conversation history: {error_msg}", 
                "error"
            )
            return CommandResult(False, f"Save failed: {error_msg}")