"""
Chunk management command handlers for listing, expanding, and adding chunks.
"""

from typing import List

from ragger.ui.resources import Emojis, format_error_message
from ragger.ui.terminal.text_utils import get_wrapped_text_in_box
from ragger.interfaces.command_handlers import CommandHandler, CommandResult
from ragger.config import CHUNK_PREVIEW_LENGTH_MEDIUM, CHUNK_SEPARATOR_LINE_LENGTH
from ragger.core.commands import AddCommand, Command, ExpandCommand, ExpandCustomCommand, ListChunksCommand


class ChunkDisplayHandler(CommandHandler):
    """Handles chunk display operations (list, expand)."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['ListChunksCommand', 'ExpandCommand', 'ExpandCustomCommand']
    
    def validate(self, command: Command) -> bool:
        """Validate chunk display commands."""
        if isinstance(command, (ExpandCommand, ExpandCustomCommand)):
            return command.chunk_num is not None
        return True
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute chunk display command."""
        if isinstance(command, ListChunksCommand):
            return self._handle_list_chunks(command, cli_interface)
        elif isinstance(command, ExpandCommand):
            return self._handle_expand_chunk(command, cli_interface)
        elif isinstance(command, ExpandCustomCommand):
            return self._handle_expand_custom_chunk(command, cli_interface)
        else:
            return CommandResult(False, f"Unsupported command type: {command.__class__.__name__}")
    
    def _handle_list_chunks(self, command: ListChunksCommand, cli_interface) -> CommandResult:
        """Handle listing custom chunks."""
        result = cli_interface.rag_service.get_custom_chunks()
        
        if result['chunks_count'] == 0:
            cli_interface.ui.add_response(f"\n{Emojis.INFO} No custom chunks available yet.", "info")
            return CommandResult(True, "No chunks to display")
        
        # Format header
        header = f"\n{Emojis.CHUNKS} Custom chunks ({result['chunks_count']}):"
        cli_interface.ui.add_response(header, "chunks_header")
        
        # Format all chunks into a single text block
        all_chunks_text = []
        for i, chunk in enumerate(result['chunks'], 1):
            preview = chunk.page_content.strip()[:CHUNK_PREVIEW_LENGTH_MEDIUM]
            if len(chunk.page_content) > CHUNK_PREVIEW_LENGTH_MEDIUM:
                preview += "..."
            source = chunk.metadata.get("source", "Unknown source")
            chunk_text = f"Chunk #{i}: {preview}\nSource: {source}"
            all_chunks_text.append(chunk_text)
            
            # Add separator between chunks if not the last one
            if i < result['chunks_count']:
                all_chunks_text.append("-" * CHUNK_SEPARATOR_LINE_LENGTH)
        
        # Format all chunks in a dashed box
        box_width = cli_interface.ui.app.output.get_size().columns - 2
        boxed_chunks = get_wrapped_text_in_box(
            "\n".join(all_chunks_text),
            box_width,
            "",
            border_style="dashed"
        )
        
        # Add the boxed chunks
        cli_interface.ui.add_response(boxed_chunks, "chunk")
        return CommandResult(True, f"Displayed {result['chunks_count']} custom chunks")
    
    def _handle_expand_chunk(self, command: ExpandCommand, cli_interface) -> CommandResult:
        """Handle expanding a chunk from search results."""
        if command.chunk_num is None:
            cli_interface.ui.add_response(format_error_message("Please specify a chunk number to expand"), "error")
            return CommandResult(False, "Missing chunk number")
        
        # Expand the chunk from search results (not custom chunks)
        result = cli_interface.rag_service.expand_chunk(command.chunk_num, command.context_size, from_custom=False)
        
        if result['success']:
            # Format header
            header = f"\n{Emojis.CHUNKS} Expanded search chunk #{command.chunk_num}:"
            if result.get('source_path'):
                source = result.get('source_path')
                header += f" (from {source})"
            
            # Add header
            cli_interface.ui.add_response(header, "chunks_header")
            
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
            return CommandResult(True, f"Expanded search chunk #{command.chunk_num}")
        else:
            error_msg = result.get('error', 'Unknown error')
            cli_interface.ui.add_response(format_error_message(f"Failed to expand search chunk: {error_msg}"), "error")
            return CommandResult(False, error_msg)
    
    def _handle_expand_custom_chunk(self, command: ExpandCustomCommand, cli_interface) -> CommandResult:
        """Handle expanding a custom chunk with context."""
        if command.chunk_num is None:
            cli_interface.ui.add_response(format_error_message("Please specify a chunk number to expand"), "error")
            return CommandResult(False, "Missing chunk number")
        
        # Expand the custom chunk
        result = cli_interface.rag_service.expand_custom_chunk(command.chunk_num, command.context_size)
        
        if result['success']:
            # Format header
            header = f"\n{Emojis.CHUNKS} Expanded custom chunk #{command.chunk_num}:"
            if result.get('source_path'):
                source = result.get('source_path')
                header += f" (from {source})"
            
            # Add header
            cli_interface.ui.add_response(header, "chunks_header")
            
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
            return CommandResult(True, f"Expanded custom chunk #{command.chunk_num}")
        else:
            error_msg = result.get('error', 'Unknown error')
            cli_interface.ui.add_response(format_error_message(f"Failed to expand custom chunk: {error_msg}"), "error")
            return CommandResult(False, error_msg)


class AddCommandHandler(CommandHandler):
    """Handles adding chunks to custom chunks list."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['AddCommand']
    
    def validate(self, command: Command) -> bool:
        """Validate add command."""
        if isinstance(command, AddCommand):
            return bool(command.chunk_nums)  # Must have at least one chunk number
        return True
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute add command."""
        if isinstance(command, AddCommand):
            return self._handle_add_chunks(command, cli_interface)
        else:
            return CommandResult(False, f"Unsupported command type: {command.__class__.__name__}")
    
    def _handle_add_chunks(self, command: AddCommand, cli_interface) -> CommandResult:
        """Handle adding chunk(s) to custom chunks."""
        if not command.chunk_nums:
            cli_interface.ui.add_response(format_error_message("Please specify chunk number(s) to add"), "error")
            return CommandResult(False, "No chunk numbers provided")
        
        # First check if we have recent chunks
        if not cli_interface.rag_service.retrieved_chunks_history:
            cli_interface.ui.add_response(format_error_message("No retrieved chunks available yet"), "error")
            return CommandResult(False, "No retrieved chunks available")
        
        # Find the most recent chunks
        recent_chunks = None
        for chunks in reversed(cli_interface.rag_service.retrieved_chunks_history):
            if chunks:
                recent_chunks = chunks
                break
        
        if not recent_chunks:
            cli_interface.ui.add_response(
                f"\n{Emojis.ERROR} No chunks available from previous queries.", 
                "error"
            )
            return CommandResult(False, "No chunks from previous queries")
        
        # Process each chunk number
        added_chunks = []
        failed_chunks = []
        
        for chunk_num in command.chunk_nums:
            # Check if the chunk number is valid
            if chunk_num < 1 or chunk_num > len(recent_chunks):
                failed_chunks.append(f"#{chunk_num} (invalid: range 1-{len(recent_chunks)})")
                continue
            
            # Get the chunk directly
            chunk = recent_chunks[chunk_num - 1]
            if chunk:
                add_result = cli_interface.rag_service.add_to_custom_chunks(chunk)
                if add_result['success']:
                    added_chunks.append(chunk_num)
                else:
                    failed_chunks.append(f"#{chunk_num} ({add_result.get('error', 'Unknown error')})")
            else:
                failed_chunks.append(f"#{chunk_num} (could not retrieve)")
        
        # Display results
        success_msg = ""
        if added_chunks:
            if len(added_chunks) == 1:
                cli_interface.ui.add_response(
                    f"\n{Emojis.CHECK} Added chunk #{added_chunks[0]} to your custom chunks list!", 
                    "info"
                )
                success_msg = f"Added chunk #{added_chunks[0]}"
            else:
                chunk_list = ', '.join(f"#{num}" for num in added_chunks)
                cli_interface.ui.add_response(
                    f"\n{Emojis.CHECK} Added {len(added_chunks)} chunks ({chunk_list}) to your custom chunks list!", 
                    "info"
                )
                success_msg = f"Added {len(added_chunks)} chunks"
            
            cli_interface.ui.add_response(
                f"You now have {len(getattr(cli_interface.rag_service, 'custom_chunks', []))} chunks in your personal archive.", 
                "info"
            )
            cli_interface.ui.add_response("Use 'lc' to list all your custom chunks.", "info")
            
            # Update chunk counts
            cli_interface.ui.set_chunk_counts(
                len(getattr(cli_interface.rag_service, 'custom_chunks', [])),
                cli_interface.ui.state.retrieved_chunks_count
            )
        
        if failed_chunks:
            failed_list = ', '.join(failed_chunks)
            cli_interface.ui.add_response(format_error_message(f"Failed to add chunks: {failed_list}"), "error")
            
            if not added_chunks:
                return CommandResult(False, f"Failed to add any chunks: {failed_list}")
        
        return CommandResult(True, success_msg if success_msg else "Processed chunk additions")