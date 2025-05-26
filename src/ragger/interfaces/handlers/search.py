"""
Search command handlers for vector database search operations.
"""

from typing import List

from ragger.core.commands import Command, SearchCommand
from ragger.interfaces.command_handlers import CommandHandler, CommandResult
from ragger.ui.resources import Emojis, format_error_message
from ragger.ui.terminal import format_chunk_display
from ragger.ui.terminal.text_utils import get_wrapped_text_in_box


class SearchCommandHandler(CommandHandler):
    """Handles search operations."""
    
    def get_supported_command_types(self) -> List[str]:
        return ['SearchCommand']
    
    def validate(self, command: Command) -> bool:
        """Validate search command."""
        if isinstance(command, SearchCommand):
            return command.query is not None and command.query.strip()
        return True
    
    def _execute(self, command: Command, cli_interface) -> CommandResult:
        """Execute search command."""
        if isinstance(command, SearchCommand):
            return self._handle_search(command, cli_interface)
        else:
            return CommandResult(False, f"Unsupported command type: {command.__class__.__name__}")
    
    def _handle_search(self, command: SearchCommand, cli_interface) -> CommandResult:
        """Handle search command."""
        if not command.query:
            cli_interface.ui.add_response(format_error_message("Please provide a search query"), "error")
            return CommandResult(False, "Missing search query")
        
        cli_interface.ui.add_response(f"\n{Emojis.SEARCH} Searching for: {command.query}", "info")
        
        # Use direct vector search (ignores custom chunks)
        search_result = cli_interface.rag_service._perform_search(command.query)
        
        if not search_result.get('success'):
            error_msg = search_result.get('error', 'Unknown error')
            cli_interface.ui.add_response(format_error_message(f"Search failed: {error_msg}"), "error")
            return CommandResult(False, f"Search failed: {error_msg}")
        
        # Display results
        retrieved_docs = search_result.get('retrieved_docs', [])
        if retrieved_docs:
            # Store retrieved chunks in history for later access by expand/add commands
            cli_interface.rag_service.retrieved_chunks_history.append(list(retrieved_docs))
            
            # Update chunk counts
            cli_interface.ui.set_chunk_counts(
                len(getattr(cli_interface.rag_service, 'custom_chunks', [])),
                len(retrieved_docs)
            )
            
            # Display chunks if not hidden
            if not cli_interface.hide_chunks:
                # Format retrieved chunks header
                cli_interface.ui.add_response(f"\n{Emojis.CHUNKS} Search results:", "chunks_header")
                
                # Get the terminal width
                box_width = cli_interface.ui.app.output.get_size().columns - 2
                
                # Collect all chunk information
                all_chunks_text = []
                
                # Format each chunk
                for i, doc in enumerate(retrieved_docs, 1):
                    if cli_interface.full_chunks:
                        # Full chunk details
                        chunk_info = format_chunk_display(doc, i, True, box_width-2)
                    else:
                        # Simple chunk preview with formatting
                        chunk_info = format_chunk_display(doc, i, False, box_width-2)
                    
                    all_chunks_text.append(chunk_info)
                    
                    # Add separator between chunks if not the last one
                    if i < len(retrieved_docs):
                        all_chunks_text.append("-" * (box_width - 10))
                
                # Format all chunks in a dashed box
                boxed_chunks = get_wrapped_text_in_box(
                    "\n".join(all_chunks_text),
                    box_width,
                    "",
                    border_style="dashed"
                )
                
                # Add the boxed chunks
                cli_interface.ui.add_response(boxed_chunks, "chunk")
            
            return CommandResult(True, f"Found {len(retrieved_docs)} results for: {command.query}")
        else:
            cli_interface.ui.add_response(f"\n{Emojis.WARNING} No matching chunks found.", "warning")
            return CommandResult(True, f"No results found for: {command.query}")