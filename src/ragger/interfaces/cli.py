"""
Copyright (c) 2025 Jakob Bolliger

This file is part of Ragger.

This program is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
The full text of the license can be found in the
LICENSE file in the root directory of this source tree.

CLI interface for the Ragger application.

This module provides a Command Line Interface (CLI) to the Ragger application,
connecting the core RAG functionality with a terminal-based UI.
"""
import sys
from typing import Dict, Any, Optional

from ragger.ui.terminal import RaggerUI, InputMode
from ragger.ui.terminal.text_utils import get_wrapped_text_in_box
from ragger.ui.resources import Emojis, BOXED_BANNER
from ragger.ui.terminal import format_chunk_display


class CliInterface:
    """CLI interface for the Ragger application."""
    
    def __init__(self, rag_service, command_manager, debug_mode=False):
        """Initialize the CLI interface.
        
        Args:
            rag_service: RAG service to use
            command_manager: Command manager to use
            debug_mode: Enable debug mode for troubleshooting UI issues
        """
        self.rag_service = rag_service
        self.command_manager = command_manager
        
        # Settings with defaults
        self.hide_chunks = False
        self.full_chunks = False
        self.auto_save = False
        self.debug_mode = debug_mode
        
        # Create UI instance
        self.ui = RaggerUI(
            on_query=self._handle_query,
            on_command=self._handle_command,
            debug_mode=debug_mode  # Pass through debug mode setting
        )
    
    def _handle_query(self, query: str):
        """Handle a query from the UI."""
        if not query.strip():
            return
            
        # Format and display the user prompt in a box
        box_width = self.ui.app.output.get_size().columns - 2
        formatted_text = f"\n{Emojis.USER} Prompt:"
        self.ui.add_response(formatted_text, "prompt_header") 
        
        # Create a boxed prompt and add it as a response
        boxed_text = get_wrapped_text_in_box(
            query, 
            box_width, 
            "", 
            border_style="solid"
        )
        self.ui.add_response(boxed_text, "prompt")
            
        # Display search message
        self.ui.add_response(f"\n{Emojis.SEARCH} Searching knowledge base...", "info")
        
        # Query the RAG service
        result = self.rag_service.query(query)
        
        # Update chunk counts
        if result.get('retrieved_docs'):
            # Store retrieved chunks in history for later access by expand/add commands
            self.rag_service.retrieved_chunks_history.append(list(result.get('retrieved_docs', [])))
            
            self.ui.set_chunk_counts(
                len(getattr(self.rag_service, 'custom_chunks', [])),
                len(result.get('retrieved_docs', []))
            )
            
            # Display retrieved chunks if not hidden
            if not self.hide_chunks and result.get('retrieved_docs'):
                # Format retrieved chunks
                chunk_text = f"\n{Emojis.CHUNKS} Retrieved chunks:"
                self.ui.add_response(chunk_text, "chunks_header")
                
                # Format all chunks in a box
                all_chunks_text = []
                
                # Add each chunk with formatting
                for i, doc in enumerate(result.get('retrieved_docs', []), 1):
                    if self.full_chunks:
                        # Full chunk details
                        chunk_info = format_chunk_display(doc, i, True, box_width-2)
                    else:
                        # Simple chunk preview
                        chunk_info = format_chunk_display(doc, i, False, box_width-2)
                        
                    all_chunks_text.append(chunk_info)
                    
                    # Add separator between chunks if not the last one
                    if i < len(result.get('retrieved_docs', [])):
                        all_chunks_text.append("-" * (box_width - 10))
                
                # Create a boxed representation of all chunks
                boxed_chunks = get_wrapped_text_in_box(
                    "\n".join(all_chunks_text),
                    box_width,
                    "",
                    border_style="dashed"
                )
                
                # Add the formatted chunks
                self.ui.add_response(boxed_chunks, "chunk")
        
        # Display the response in a box
        if result.get('success', False):
            # Format the model name nicely
            model_name = self.rag_service.llm_model.capitalize()
            response_header = f"\n{Emojis.ROBOT} {model_name}:"
            self.ui.add_response(response_header, "response_header")
            
            # Box the response
            boxed_response = get_wrapped_text_in_box(
                result['answer'],
                box_width,
                "",
                border_style="solid"
            )
            
            # Add the boxed response
            self.ui.add_response(boxed_response, "response")
        else:
            # Handle error case
            self.ui.add_response(f"\n{Emojis.ERROR} Error: {result.get('error', 'Unknown error')}", "error")
            
            # Display error response if available
            if result.get('answer'):
                model_name = self.rag_service.llm_model.capitalize()
                response_header = f"\n{Emojis.ROBOT} {model_name}:"
                self.ui.add_response(response_header, "response_header")
                
                # Box the error response
                boxed_response = get_wrapped_text_in_box(
                    result['answer'],
                    box_width,
                    "",
                    border_style="solid"
                )
                
                # Add the boxed response
                self.ui.add_response(boxed_response, "response")
                
        # Display warning if no chunks were retrieved
        if not result.get('retrieved_docs'):
            self.ui.add_response(f"\n{Emojis.WARNING} No chunks were successfully retrieved.", "warning")
    
    def _handle_command(self, cmd_text: str):
        """Handle a command from the UI."""
        if not cmd_text.strip():
            return
            
        # First handle special commands that don't need the command parser
        cmd_lower = cmd_text.lower().strip()
        
        # Exit commands
        if cmd_lower in ['exit', 'quit', 'q']:
            # Handle auto-save if enabled
            if self.auto_save and self.rag_service.conversation_history:
                self.ui.add_response("Auto-saving conversation history...", "info")
                result = self.rag_service.save_history()
                if result['success']:
                    self.ui.add_response(f"Conversation history saved to {result['filepath']}", "info")
                else:
                    self.ui.add_response(f"Error saving conversation history: {result.get('error', 'Unknown error')}", "error")
            
            self.ui.add_response("Goodbye!", "info")
            # Force update the display to show the goodbye message
            self.ui._update_output_buffer()
            
            # Properly exit the application
            sys.exit(0)
            
        # Help command
        elif cmd_lower in ['help', 'h', '?']:
            # Show the help information
            self.ui._show_help()
            return
            
        # Parse the command
        command = self.command_manager.parse_input(cmd_text)
        
        if command:
            # Process the command based on its type
            self._process_command(command, cmd_text)
        else:
            # Not a recognized command
            self.ui.add_response(f"Unknown command: '{cmd_text.strip()}'", "error")
    
    def _process_command(self, command, cmd_text: str):
        """Process a parsed command."""
        # Handle different command types
        command_type = command.__class__.__name__
        
        match command_type:
            case "HistoryCommand":
                # Format and display history
                if not self.rag_service.conversation_history:
                    self.ui.add_response(f"\n{Emojis.INFO} No conversation history yet.", "info")
                    return
                    
                # Add the header
                self.ui.add_response(f"\n{Emojis.HISTORY} Conversation History:", "info")
                
                # Get the terminal width
                box_width = self.ui.app.output.get_size().columns - 2
                
                # Format each turn in history
                for i, (prompt, response) in enumerate(self.rag_service.conversation_history, 1):
                    # Add turn header
                    self.ui.add_response(f"\n--- Turn {i} ---", "info")
                    
                    # Format and display the prompt
                    prompt_header = f"{Emojis.USER} Prompt:"
                    boxed_prompt = get_wrapped_text_in_box(
                        prompt,
                        box_width,
                        prompt_header,
                        border_style="solid"
                    )
                    self.ui.add_response(boxed_prompt, "prompt")
                    
                    # Format and display the response
                    model_name = self.rag_service.llm_model.capitalize()
                    response_header = f"{Emojis.ROBOT} {model_name}:"
                    boxed_response = get_wrapped_text_in_box(
                        response,
                        box_width,
                        response_header,
                        border_style="solid"
                    )
                    self.ui.add_response(boxed_response, "response")
                    
                    # Add separator between turns if not the last one
                    if i < len(self.rag_service.conversation_history):
                        self.ui.add_response("\n" + "-" * box_width, "info")
            
            case "ClearCommand":
                # Clear history
                self.rag_service.clear_history(preserve_custom_chunks=False)
                self.ui.add_response("All history and custom chunks cleared!", "info")
                self.ui.set_chunk_counts(0, 0)
                
            case "ClearHistoryCommand":
                # Clear only conversation history
                self.rag_service.clear_history(preserve_custom_chunks=True)
                self.ui.add_response("Conversation history cleared!", "info")
                
            case "ClearChunksCommand":
                # Clear only custom chunks
                result = self.rag_service.clear_custom_chunks()
                self.ui.add_response("Custom chunks cleared!", "info")
                self.ui.set_chunk_counts(0, self.ui.state.retrieved_chunks_count)
                
            case "SaveCommand":
                # Save history
                if not self.rag_service.conversation_history:
                    self.ui.add_response("No conversation history to save yet.", "warning")
                else:
                    result = self.rag_service.save_history()
                    if result['success']:
                        self.ui.add_response(f"Conversation history saved to {result['filepath']}", "info")
                    else:
                        self.ui.add_response(f"Error saving conversation history: {result.get('error', 'Unknown error')}", "error")
                        
            case "AddCommand":
                # Add a chunk to custom chunks
                if command.chunk_num is None:
                    self.ui.add_response(f"\n{Emojis.ERROR} Please specify a chunk number to add.", "error")
                    return
                    
                # First check if we have recent chunks
                if not self.rag_service.retrieved_chunks_history:
                    self.ui.add_response(f"\n{Emojis.ERROR} No retrieved chunks available yet.", "error")
                    return
                    
                # Find the most recent chunks
                recent_chunks = None
                for chunks in reversed(self.rag_service.retrieved_chunks_history):
                    if chunks:
                        recent_chunks = chunks
                        break
                        
                if not recent_chunks:
                    self.ui.add_response(f"\n{Emojis.ERROR} No chunks available from previous queries.", "error")
                    return
                    
                # Check if the chunk number is valid
                if command.chunk_num < 1 or command.chunk_num > len(recent_chunks):
                    self.ui.add_response(f"\n{Emojis.ERROR} Invalid chunk number. Available chunks: 1-{len(recent_chunks)}", "error")
                    return
                    
                # Get the chunk directly
                chunk = recent_chunks[command.chunk_num - 1]
                if chunk:
                    add_result = self.rag_service.add_to_custom_chunks(chunk)
                    if add_result['success']:
                        # Get preview of the chunk content for display
                        preview = chunk.page_content.strip()[:100]
                        if len(chunk.page_content) > 100:
                            preview += "..."
                        
                        # Format success message with more details
                        self.ui.add_response(f"\n{Emojis.CHECK} Added chunk #{command.chunk_num} to your custom chunks list!", "info")
                        self.ui.add_response(f"You now have {add_result.get('chunks_count', 1)} chunks in your personal archive.", "info")
                        self.ui.add_response(f"\nPreview: \"{preview}\"", "info")
                        self.ui.add_response(f"\nUse 'ap {add_result.get('chunk_num', command.chunk_num)}' to add this chunk to your prompt when needed.", "info")
                        self.ui.add_response("Use 'lc' to list all your custom chunks.", "info")
                        
                        # Update chunk counts
                        self.ui.set_chunk_counts(
                            len(getattr(self.rag_service, 'custom_chunks', [])),
                            self.ui.state.retrieved_chunks_count
                        )
                    else:
                        self.ui.add_response(f"\n{Emojis.ERROR} Error adding chunk: {add_result.get('error', 'Unknown error')}", "error")
                else:
                    self.ui.add_response(f"\n{Emojis.ERROR} Error: Could not retrieve chunk for adding.", "error")
                    
            case "ExpandCommand":
                # Expand a chunk with context
                if command.chunk_num is None:
                    self.ui.add_response(f"\n{Emojis.ERROR} Please specify a chunk number to expand.", "error")
                    return
                    
                result = self.rag_service.expand_chunk(command.chunk_num, command.context_size)
                if result['success']:
                    # Format header
                    header = f"\n{Emojis.CHUNKS} Expanded chunk #{command.chunk_num}:"
                    if result.get('source_path'):
                        source = result.get('source_path')
                        header += f" (from {source})"
                    
                    # Add header
                    self.ui.add_response(header, "chunks_header")
                    
                    # Format content in a dashed box
                    box_width = self.ui.app.output.get_size().columns - 2
                    boxed_text = get_wrapped_text_in_box(
                        result['expanded_context'],
                        box_width,
                        "",
                        border_style="dashed"
                    )
                    
                    # Add the boxed content
                    self.ui.add_response(boxed_text, "chunk")
                else:
                    self.ui.add_response(f"\n{Emojis.ERROR} Error expanding chunk: {result.get('error', 'Unknown error')}", "error")
                    
            case "ListChunksCommand":
                # List custom chunks
                result = self.rag_service.get_custom_chunks()
                
                if result['chunks_count'] == 0:
                    self.ui.add_response(f"\n{Emojis.INFO} No custom chunks available yet.", "info")
                else:
                    # Format header
                    header = f"\n{Emojis.CHUNKS} Custom chunks ({result['chunks_count']}):"
                    self.ui.add_response(header, "chunks_header")
                    
                    # Format all chunks into a single text block
                    all_chunks_text = []
                    for i, chunk in enumerate(result['chunks'], 1):
                        preview = chunk.page_content.strip()[:100]
                        if len(chunk.page_content) > 100:
                            preview += "..."
                        source = chunk.metadata.get("source", "Unknown source")
                        chunk_text = f"Chunk #{i}: {preview}\nSource: {source}"
                        all_chunks_text.append(chunk_text)
                        
                        # Add separator between chunks if not the last one
                        if i < result['chunks_count']:
                            all_chunks_text.append("-" * 40)
                    
                    # Format all chunks in a dashed box
                    box_width = self.ui.app.output.get_size().columns - 2
                    boxed_chunks = get_wrapped_text_in_box(
                        "\n".join(all_chunks_text),
                        box_width,
                        "",
                        border_style="dashed"
                    )
                    
                    # Add the boxed chunks
                    self.ui.add_response(boxed_chunks, "chunk")
                    
            case "AddPromptCommand":
                # Add a custom chunk to prompt
                if command.chunk_num is None:
                    self.ui.add_response(f"\n{Emojis.ERROR} Please specify a custom chunk number.", "error")
                    return
                    
                result = self.rag_service.expand_custom_chunk(command.chunk_num, command.context_size)
                if result['success']:
                    self.ui.add_response(f"\n{Emojis.INFO} Added custom chunk #{command.chunk_num} to prompt context:", "info")
                    
                    # Format content in a dashed box
                    box_width = self.ui.app.output.get_size().columns - 2
                    boxed_text = get_wrapped_text_in_box(
                        result['expanded_context'],
                        box_width,
                        "",
                        border_style="dashed"
                    )
                    
                    # Add the boxed content
                    self.ui.add_response(boxed_text, "chunk")
                    self.ui.add_response(f"\n{Emojis.INFO} Your next query will include this context.", "info")
                else:
                    self.ui.add_response(f"\n{Emojis.ERROR} Error: {result.get('error', 'Unknown error')}", "error")
                    
            case "SearchCommand":
                # Handle search command
                if not command.query:
                    self.ui.add_response("Please provide a search query.", "error")
                    return
                
                self.ui.add_response(f"\n{Emojis.SEARCH} Searching for: {command.query}", "info")
                
                # Use the RAG service to perform the search
                result = self.rag_service.query(command.query, use_history=False)
                
                # Display results
                if result.get('retrieved_docs'):
                    # Store retrieved chunks in history for later access by expand/add commands
                    self.rag_service.retrieved_chunks_history.append(list(result.get('retrieved_docs', [])))
                    
                    # Update chunk counts
                    self.ui.set_chunk_counts(
                        len(getattr(self.rag_service, 'custom_chunks', [])),
                        len(result.get('retrieved_docs', []))
                    )
                    
                    # Display chunks if not hidden
                    if not self.hide_chunks:
                        # Format retrieved chunks header
                        self.ui.add_response(f"\n{Emojis.CHUNKS} Search results:", "chunks_header")
                        
                        # Get the terminal width
                        box_width = self.ui.app.output.get_size().columns - 2
                        
                        # Collect all chunk information
                        all_chunks_text = []
                        
                        # Format each chunk
                        for i, doc in enumerate(result.get('retrieved_docs', []), 1):
                            if self.full_chunks:
                                # Full chunk details
                                chunk_info = format_chunk_display(doc, i, True, box_width-2)
                            else:
                                # Simple chunk preview with formatting
                                chunk_info = format_chunk_display(doc, i, False, box_width-2)
                            
                            all_chunks_text.append(chunk_info)
                            
                            # Add separator between chunks if not the last one
                            if i < len(result.get('retrieved_docs', [])):
                                all_chunks_text.append("-" * (box_width - 10))
                        
                        # Format all chunks in a dashed box
                        boxed_chunks = get_wrapped_text_in_box(
                            "\n".join(all_chunks_text),
                            box_width,
                            "",
                            border_style="dashed"
                        )
                        
                        # Add the boxed chunks
                        self.ui.add_response(boxed_chunks, "chunk")
                else:
                    self.ui.add_response(f"\n{Emojis.WARNING} No matching chunks found.", "warning")
                    
            case "SetChunksCommand":
                # Set number of chunks to retrieve
                if command.num_chunks is None:
                    self.ui.add_response(f"\n{Emojis.ERROR} Please specify the number of chunks to retrieve.", "error")
                    return
                    
                # Call the RAG service to set the number of chunks
                result = self.rag_service.set_chunks(command.num_chunks)
                
                if result['success']:
                    # Show success message with previous and new values
                    previous = result['previous_value']
                    current = result['current_value']
                    self.ui.add_response(f"\n{Emojis.CHECK} Number of chunks to retrieve changed from {previous} to {current}.", "info")
                    self.ui.add_response(f"Use the '-sc' or '--set-chunks' command-line option to set this value when starting the application.", "info")
                else:
                    # Show error message
                    self.ui.add_response(f"\n{Emojis.ERROR} Error: {result.get('error', 'Unknown error')}", "error")
                    
            case "GetChunksCommand":
                # Display current chunks setting
                num_chunks = self.rag_service.num_chunks
                self.ui.add_response(f"\n{Emojis.CHUNKS} Current number of chunks to retrieve: {num_chunks}", "info")
                self.ui.add_response(f"Use 'sc <number>' to change this setting.", "info")
                self.ui.add_response(f"Use the '-sc' or '--set-chunks' command-line option to set this value when starting the application.", "info")
                
            case "PromptCommand":
                # Handle prompt command
                if not command.query:
                    self.ui.add_response("Please provide a prompt query.", "error")
                    return
                    
                # Call the _handle_query method with the prompt text
                self._handle_query(command.query)
                
            case _:
                # Handle unknown command type
                self.ui.add_response(f"\n{Emojis.ERROR} Unknown command type: {command_type}", "error")
    
    def interactive_session(self, hide_chunks: bool = False, full_chunks: bool = False, auto_save: bool = False):
        """Run an interactive CLI session."""
        # Store settings
        self.hide_chunks = hide_chunks
        self.full_chunks = full_chunks
        self.auto_save = auto_save
        
        banner = BOXED_BANNER
        ui_created = False
        try:
            # Step 1: Initialize UI with welcome messages
            # Add initial banner
            self.ui.add_response(banner, "info")
            
            # Display initial instructions right away - no loading message since we show it in console
            self.ui.add_response("\nPress Tab to switch modes or F2 for help.", "info")
            
            # Force an update of the UI to show these messages right away
            self.ui._update_output_buffer()
            self.ui.app.invalidate()
            ui_created = True
            
            # Set initial chunk counts
            self.ui.set_chunk_counts(
                len(getattr(self.rag_service, 'custom_chunks', [])),
                0  # No chunks retrieved initially
            )

            if self.rag_service.vectorstore:
                self.ui.add_response(f"\n{Emojis.CHECK} Vector database loaded successfully from:"
                                     f" {self.rag_service.db_path}", "info")
            else:
                self.ui.add_response("Error loading vector database.", "error")
                self.ui.add_response("Please check the path and try again.", "info")
                return
        except Exception as e:
            print(f"Error during UI startup: {str(e)}")
            # If the UI failed to create properly, handle gracefully
            if not ui_created:
                print(banner)
                print("\nError initializing UI. Check your terminal configuration.")
                return
            
        # Display success and model info
        self.ui.add_response(f"\n{Emojis.ROBOT} Using LLM model: {self.rag_service.llm_model}", "info")
        self.ui.add_response(f"\n{Emojis.CHUNKS} Number of chunks to retrieve: {self.rag_service.num_chunks}", "info")
        self.ui.add_response(f"\n{Emojis.INFO} Note: The first query may take longer as the model needs to load.", "info")

        # Run the UI
        try:
            self.ui.run()
            
            # Handle auto-save if enabled when exiting
            if auto_save and self.rag_service.conversation_history:
                result = self.rag_service.save_history()
                if result['success']:
                    print(f"\nConversation history saved to {result['filepath']}")
                else:
                    print(f"\nError saving conversation history: {result.get('error', 'Unknown error')}")
                    
        except KeyboardInterrupt:
            # Handle auto-save on keyboard interrupt
            if auto_save and self.rag_service.conversation_history:
                try:
                    result = self.rag_service.save_history()
                    if result['success']:
                        print(f"\nConversation history saved to {result['filepath']}")
                    else:
                        print(f"\nError saving conversation history: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"\nError saving conversation history: {str(e)}")
                    
            print("\nGoodbye!")