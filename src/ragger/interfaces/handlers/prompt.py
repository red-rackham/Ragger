"""
Prompt command handlers for direct LLM interaction.
"""

from typing import List

from ragger.ui.resources import Emojis, format_error_message
from ragger.ui.terminal.text_utils import get_wrapped_text_in_box
from ragger.core.commands import AddPromptCommand, Command, PromptCommand
from ragger.interfaces.command_handlers import CommandHandler, CommandResult


class PromptCommandHandler(CommandHandler):
    """Handles direct prompt commands."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['PromptCommand']
    
    def validate(self, command: Command) -> bool:
        """Validate prompt command."""
        if isinstance(command, PromptCommand):
            return command.query is not None and command.query.strip()
        return True
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute prompt command."""
        if isinstance(command, PromptCommand):
            return self._handle_prompt(command, cli_interface)
        else:
            return CommandResult(False, f"Unsupported command type: {command.__class__.__name__}")
    
    def _handle_prompt(self, command: PromptCommand, cli_interface) -> CommandResult:
        """Handle direct prompt execution."""
        if not command.query:
            cli_interface.ui.add_response(format_error_message("Please provide a prompt query"), "error")
            return CommandResult(False, "Missing prompt query")
        
        # Call the _handle_query method with the prompt text
        cli_interface._handle_query(command.query)
        return CommandResult(True, f"Executed prompt: {command.query[:50]}...")


class AddPromptCommandHandler(CommandHandler):
    """Handles adding custom chunks to prompts."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['AddPromptCommand']
    
    def validate(self, command: Command) -> bool:
        """Validate add prompt command."""
        if isinstance(command, AddPromptCommand):
            return command.chunk_num is not None
        return True
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute add prompt command."""
        if isinstance(command, AddPromptCommand):
            return self._handle_add_prompt(command, cli_interface)
        else:
            return CommandResult(False, f"Unsupported command type: {command.__class__.__name__}")
    
    def _handle_add_prompt(self, command: AddPromptCommand, cli_interface) -> CommandResult:
        """Handle adding a custom chunk to prompt."""
        if command.chunk_num is None:
            cli_interface.ui.add_response(format_error_message("Please specify a custom chunk number"), "error")
            return CommandResult(False, "Missing chunk number")
        
        # Expand the custom chunk
        result = cli_interface.rag_service.expand_custom_chunk(command.chunk_num, command.context_size)
        
        if result['success']:
            cli_interface.ui.add_response(
                f"\n{Emojis.INFO} Added custom chunk #{command.chunk_num} to prompt context:", 
                "info"
            )
            
            # Format content in a dashed box
            box_width = cli_interface.ui.app.output.get_size().columns - 2
            boxed_text = get_wrapped_text_in_box(
                result['expanded_context'],
                box_width,
                "",
                border_style="dashed"
            )
            
            # Add the boxed content
            cli_interface.ui.add_response(boxed_text, "chunk")
            cli_interface.ui.add_response(
                f"\n{Emojis.INFO} Your next query will include this context.", 
                "info"
            )
            return CommandResult(True, f"Added custom chunk #{command.chunk_num} to prompt")
        else:
            error_msg = result.get('error', 'Unknown error')
            cli_interface.ui.add_response(format_error_message(f"Failed to add chunk to prompt: {error_msg}"), "error")
            return CommandResult(False, error_msg)