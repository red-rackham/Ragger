"""
History management command handlers for displaying and clearing history.
"""

from typing import List

from ragger.ui.resources import Emojis
from ragger.ui.terminal.text_utils import get_wrapped_text_in_box
from ragger.interfaces.command_handlers import CommandHandler, CommandResult
from ragger.core.commands import (ClearChunksCommand, ClearCommand,
                                  ClearHistoryCommand, Command, HistoryCommand)


class HistoryCommandHandler(CommandHandler):
    """Handles history display command."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['HistoryCommand']
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute history display command."""
        if not cli_interface.rag_service.conversation_history:
            cli_interface.ui.add_response(f"\n{Emojis.INFO} No conversation history yet.", "info")
            return CommandResult(True, "No history to display")
        
        # Add the header
        cli_interface.ui.add_response(f"\n{Emojis.HISTORY} Conversation History:", "info")
        
        # Get the terminal width
        box_width = cli_interface.ui.app.output.get_size().columns - 2
        
        # Format each turn in history
        for i, (prompt, response) in enumerate(cli_interface.rag_service.conversation_history, 1):
            # Add turn header
            cli_interface.ui.add_response(f"\n--- Turn {i} ---", "info")
            
            # Format and display the prompt
            prompt_header = f"{Emojis.USER} Prompt:"
            boxed_prompt = get_wrapped_text_in_box(
                prompt,
                box_width,
                prompt_header,
                border_style="solid"
            )
            cli_interface.ui.add_response(boxed_prompt, "prompt")
            
            # Format and display the response
            model_name = cli_interface.rag_service.llm_model.capitalize()
            response_header = f"{Emojis.ROBOT} {model_name}:"
            boxed_response = get_wrapped_text_in_box(
                response,
                box_width,
                response_header,
                border_style="solid"
            )
            cli_interface.ui.add_response(boxed_response, "response")
            
            # Add separator between turns if not the last one
            if i < len(cli_interface.rag_service.conversation_history):
                cli_interface.ui.add_response("\n" + "-" * box_width, "info")
        
        return CommandResult(True, f"Displayed {len(cli_interface.rag_service.conversation_history)} conversation turns")


class ClearCommandHandler(CommandHandler):
    """Handles clear operations (history, chunks, all)."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['ClearCommand', 'ClearHistoryCommand', 'ClearChunksCommand']
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute clear command."""
        command_type = command.__class__.__name__
        
        if command_type == "ClearCommand":
            return self._handle_clear_all(command, cli_interface)
        elif command_type == "ClearHistoryCommand":
            return self._handle_clear_history(command, cli_interface)
        elif command_type == "ClearChunksCommand":
            return self._handle_clear_chunks(command, cli_interface)
        else:
            return CommandResult(False, f"Unsupported command type: {command_type}")
    
    def _handle_clear_all(self, command: ClearCommand, cli_interface) -> CommandResult:
        """Handle clearing all history and chunks."""
        cli_interface.rag_service.clear_history(preserve_custom_chunks=False)
        cli_interface.ui.add_response("All history and custom chunks cleared!", "info")
        cli_interface.ui.set_chunk_counts(0, 0)
        return CommandResult(True, "All data cleared")
    
    def _handle_clear_history(self, command: ClearHistoryCommand, cli_interface) -> CommandResult:
        """Handle clearing only conversation history."""
        cli_interface.rag_service.clear_history(preserve_custom_chunks=True)
        cli_interface.ui.add_response("Conversation history cleared!", "info")
        return CommandResult(True, "History cleared")
    
    def _handle_clear_chunks(self, command: ClearChunksCommand, cli_interface) -> CommandResult:
        """Handle clearing only custom chunks."""
        result = cli_interface.rag_service.clear_custom_chunks()
        cli_interface.ui.add_response("Custom chunks cleared!", "info")
        cli_interface.ui.set_chunk_counts(0, cli_interface.ui.state.retrieved_chunks_count)
        return CommandResult(True, f"Cleared {result.get('previous_count', 0)} custom chunks")