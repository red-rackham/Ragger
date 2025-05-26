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
import threading
import time
from queue import Empty, Queue
from typing import Any, Dict

from ragger.ui.terminal import RaggerUI, format_chunk_display
from ragger.utils.logging_config import get_logger, setup_logging
from ragger.ui.terminal.text_utils import get_wrapped_text_in_box
from ragger.interfaces.command_handlers import CommandHandlerRegistry
from ragger.config import LLM_RESULT_CHECK_INTERVAL, THREAD_SHUTDOWN_TIMEOUT
from ragger.ui.resources import BOXED_BANNER, Emojis, format_error_message


class CliInterface:
    """CLI interface for the Ragger application."""
    
    def __init__(self, rag_service, command_manager, debug_mode=False, verbose=False):
        """Initialize the CLI interface.
        
        Args:
            rag_service: RAG service to use
            command_manager: Command manager to use
            debug_mode: Enable debug mode for troubleshooting UI issues
            verbose: Enable verbose logging
        """
        self.rag_service = rag_service
        self.command_manager = command_manager
        
        # Settings with defaults
        self.hide_chunks = False
        self.full_chunks = False
        self.auto_save = False
        self.debug_mode = debug_mode
        self.verbose = verbose
        
        # Setup centralized logging
        setup_logging(verbose)
        self.logger = get_logger(__name__)
        
        # Thread-safe communication
        self._llm_result_queue = Queue()
        self._active_llm_threads = set()
        self._shutdown_event = threading.Event()
        
        # Command handler registry
        self.command_registry = CommandHandlerRegistry()
        
        # Create UI instance
        self.ui = RaggerUI(
            on_query=self._handle_query,
            on_command=self._handle_command,
            debug_mode=debug_mode  # Pass through debug mode setting
        )
    
    def _display_llm_result(self, result_data: Dict[str, Any], query: str, start_time: float, retrieval_time: float):
        """Display LLM result in UI thread-safely.
        
        Args:
            result_data: Result from LLM generation
            query: Original query
            start_time: Query start time
            retrieval_time: Time spent on retrieval phase
        """
        box_width = self.ui.app.output.get_size().columns - 2
        generation_time = result_data.get('generation_time', 0)
        
        if result_data.get('success', False):
            self.logger.debug("Formatting and displaying LLM response")
            # Format the model name nicely
            model_name = self.rag_service.llm_model.capitalize()
            response_header = f"\n{Emojis.ROBOT} {model_name}:"
            self.ui.add_response(response_header, "response_header")
            
            # Box the response
            boxed_response = get_wrapped_text_in_box(
                result_data['answer'],
                box_width,
                "",
                border_style="solid"
            )
            
            # Add the boxed response
            self.ui.add_response(boxed_response, "response")
            self.logger.debug("LLM response displayed")
        else:
            # Handle error case
            error_msg = result_data.get('error', 'Unknown error')
            self.logger.error(f"LLM generation failed: {error_msg}")
            self.ui.add_response(format_error_message(f"Failed to generate response: {error_msg}"), "error")
        
        total_time = time.time() - start_time
        self.logger.info(f"Complete query handling finished in {total_time:.2f}s "
                         f"(retrieval: {retrieval_time:.2f}s, generation: {generation_time:.2f}s)")
    
    def _check_llm_results(self):
        """Check for completed LLM results and display them (runs in main thread)."""
        while True:
            try:
                # Non-blocking check for results
                result_data = self._llm_result_queue.get_nowait()
                
                # Extract the context data
                query = result_data.pop('_query')
                start_time = result_data.pop('_start_time') 
                retrieval_time = result_data.pop('_retrieval_time')
                thread_id = result_data.pop('_thread_id')
                
                # Remove completed thread from tracking
                self._active_llm_threads.discard(thread_id)
                
                # Display the result
                self._display_llm_result(result_data, query, start_time, retrieval_time)
                
            except Empty:
                # No more results to process
                break
    
    def _handle_query(self, query: str):
        """Handle a query from the UI."""
        start_time = time.time()
        self.logger.info(f"Starting query handling: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        if not query.strip():
            self.logger.debug("Empty query received, returning early")
            return
            
        # Format and display the user prompt in a box
        box_width = self.ui.app.output.get_size().columns - 2
        formatted_text = f"\n{Emojis.USER} Prompt:"
        self.ui.add_response(formatted_text, "prompt_header") 
        self.logger.debug("Added prompt header to UI")
        
        # Create a boxed prompt and add it as a response
        boxed_text = get_wrapped_text_in_box(
            query, 
            box_width, 
            "", 
            border_style="solid"
        )
        self.ui.add_response(boxed_text, "prompt")
        self.logger.debug("Added boxed prompt to UI")
            
        # Phase 1: Retrieve chunks and show them immediately
        # Show appropriate message based on chunk behavior
        if self.rag_service.combine_chunks:
            self.ui.add_response(f"\n{Emojis.SEARCH} Searching knowledge base and including custom chunks...", "info")
        elif self.rag_service.ignore_custom:
            self.ui.add_response(f"\n{Emojis.SEARCH} Searching knowledge base (ignoring custom chunks)...", "info")
        elif self.rag_service.custom_chunks:
            self.ui.add_response(f"\n{Emojis.CHUNKS} Using custom chunks...", "info")
        else:
            self.ui.add_response(f"\n{Emojis.SEARCH} Searching knowledge base...", "info")
            
        self.logger.info("Phase 1: Starting chunk selection")
        
        # Get chunks using smart selection
        retrieval_start = time.time()
        retrieval_result = self.rag_service.search_only(query)
        retrieval_time = time.time() - retrieval_start
        self.logger.info(f"Chunk selection completed in {retrieval_time:.2f}s (mode: {retrieval_result.get('mode', 'unknown')})")
        
        if not retrieval_result.get('success', False):
            error_msg = retrieval_result.get('error', 'Unknown error')
            self.logger.error(f"Retrieval failed: {error_msg}")
            self.ui.add_response(format_error_message(f"Search operation failed: {error_msg}"), "error")
            return
        
        retrieved_docs = retrieval_result.get('retrieved_docs', [])
        mode = retrieval_result.get('mode', 'unknown')
        self.logger.info(f"Selected {len(retrieved_docs)} chunks using mode: {mode}")
        
        # Update chunk counts and show retrieved chunks immediately
        if retrieved_docs:
            # Store retrieved chunks in history for later access by expand/add commands
            self.rag_service.retrieved_chunks_history.append(list(retrieved_docs))
            
            # Update UI counts based on what was actually used
            search_count = len(retrieval_result.get('search_results', []))
            self.ui.set_chunk_counts(
                len(getattr(self.rag_service, 'custom_chunks', [])),
                search_count
            )
            self.logger.debug(f"Updated chunk counts in UI")
            
            # Display chunks if not hidden
            if not self.hide_chunks:
                self.logger.debug("Formatting and displaying chunks")
                
                # Choose appropriate header based on mode
                if mode == 'custom_only':
                    chunk_text = f"\n{Emojis.CHUNKS} Custom chunks:"
                elif mode == 'combine':
                    chunk_text = f"\n{Emojis.CHUNKS} Combined chunks (custom + search):"
                elif mode == 'search_only':
                    chunk_text = f"\n{Emojis.CHUNKS} Search results (custom ignored):"
                else:
                    chunk_text = f"\n{Emojis.CHUNKS} Retrieved chunks:"
                self.ui.add_response(chunk_text, "chunks_header")
                
                # Format all chunks in a box
                all_chunks_text = []
                
                # Add each chunk with formatting
                for i, doc in enumerate(retrieved_docs, 1):
                    if self.full_chunks:
                        # Full chunk details
                        chunk_info = format_chunk_display(doc, i, True, box_width-2)
                    else:
                        # Simple chunk preview
                        chunk_info = format_chunk_display(doc, i, False, box_width-2)
                        
                    all_chunks_text.append(chunk_info)
                    
                    # Add separator between chunks if not the last one
                    if i < len(retrieved_docs):
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
                chunks_display_time = time.time() - retrieval_start
                self.logger.info(f"Chunks displayed after {chunks_display_time:.2f}s from retrieval start")
            else:
                self.logger.debug("Chunks hidden by user setting")
        else:
            self.logger.warning("No chunks were retrieved")
            self.ui.add_response(f"\n{Emojis.WARNING} No chunks were retrieved.", "warning")
        
        phase1_time = time.time() - start_time
        self.logger.info(f"Phase 1 (retrieval + display) completed in {phase1_time:.2f}s")
        
        # Phase 2: Generate LLM response (if LLM model available)
        if not self.rag_service.llm_model:
            self.logger.info("No LLM model available, showing search-only messages")
            self.ui.add_response(f"\n{Emojis.WARNING} No LLM model loaded - search mode only!", "warning")
            self.ui.add_response(f"\n{Emojis.INFO} Use search commands instead:", "info")
            self.ui.add_response(f"   {Emojis.BULLET} /search <your query>  - Search the knowledge base", "info")
            self.ui.add_response(f"   {Emojis.BULLET} /chunks <number>     - Set number of results", "info")
            self.ui.add_response(f"   {Emojis.BULLET} /expand <chunk_num>  - View full chunk content", "info")
            self.ui.add_response(f"\n{Emojis.TAB} Switch to search mode with Tab key, then try your query!", "info")
            total_time = time.time() - start_time
            self.logger.info(f"Query handling completed (search-only) in {total_time:.2f}s")
            return
        
        # Show generating message
        self.logger.info("Phase 2: Starting LLM generation")
        self.ui.add_response(f"\n{Emojis.ROBOT} Generating response...", "info")
        
        # Create thread-safe LLM generation function
        def generate_llm_response():
            """Generate LLM response in background thread and queue result."""
            thread_id = threading.get_ident()
            try:
                self.logger.info(f"LLM thread {thread_id} starting generation")
                generation_result = self.rag_service.generate_response(query)
                self.logger.info(f"LLM thread {thread_id} completed generation")
                
                # Add context data for main thread
                generation_result['_query'] = query
                generation_result['_start_time'] = start_time
                generation_result['_retrieval_time'] = retrieval_time
                generation_result['_thread_id'] = thread_id
                
                # Thread-safe: Put result in queue for main thread to process
                self._llm_result_queue.put(generation_result)
                
            except Exception as e:
                # Handle unexpected errors
                self.logger.error(f"LLM thread {thread_id} failed: {str(e)}")
                error_result = {
                    'success': False,
                    'error': str(e),
                    'generation_time': 0,
                    '_query': query,
                    '_start_time': start_time,
                    '_retrieval_time': retrieval_time,
                    '_thread_id': thread_id
                }
                self._llm_result_queue.put(error_result)
            finally:
                # Clean up thread tracking
                self._active_llm_threads.discard(thread_id)
        
        # Start the LLM generation in a background thread
        generation_thread = threading.Thread(target=generate_llm_response, daemon=True)
        generation_thread.start()
        
        # Track the thread (ident is available after start())
        thread_id = generation_thread.ident
        if thread_id:
            self._active_llm_threads.add(thread_id)
        
        # Schedule periodic checks for LLM results (main thread only)
        self._schedule_result_checks()
    
    def _schedule_result_checks(self):
        """Schedule periodic checks for LLM results in the main thread."""
        def check_and_reschedule():
            # Check for any completed LLM results
            self._check_llm_results()
            
            # If there are still active threads, schedule another check
            if self._active_llm_threads and not self._shutdown_event.is_set():
                # Schedule next check in 100ms using prompt_toolkit's call_later
                try:
                    # Try to get the event loop from the application
                    loop = self.ui.app.loop
                    if loop and not loop.is_closed():
                        loop.call_later(LLM_RESULT_CHECK_INTERVAL, check_and_reschedule)
                    else:
                        # Fallback: Use app invalidate to trigger refresh
                        self.ui.app.invalidate()
                        # Manual scheduling with threading timer as fallback
                        timer = threading.Timer(LLM_RESULT_CHECK_INTERVAL, check_and_reschedule)
                        timer.daemon = True
                        timer.start()
                except Exception as e:
                    self.logger.debug(f"Fallback scheduling due to: {e}")
                    # Fallback: Use threading timer
                    timer = threading.Timer(LLM_RESULT_CHECK_INTERVAL, check_and_reschedule)
                    timer.daemon = True
                    timer.start()
        
        # Start the checking cycle if we have active threads
        if self._active_llm_threads:
            check_and_reschedule()
    
    def shutdown(self):
        """Clean shutdown of the CLI interface."""
        self.logger.info("Shutting down CLI interface")
        self._shutdown_event.set()
        
        # Wait for active threads to complete (with timeout)
        max_wait = THREAD_SHUTDOWN_TIMEOUT  # seconds timeout
        start_wait = time.time()
        
        while self._active_llm_threads and (time.time() - start_wait) < max_wait:
            time.sleep(LLM_RESULT_CHECK_INTERVAL)
            # Process any final results
            self._check_llm_results()
        
        if self._active_llm_threads:
            self.logger.warning(f"Timeout waiting for {len(self._active_llm_threads)} LLM threads to complete")
    
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
                    self.ui.add_response(f"Error saving conversation history: "
                                         f"{result.get('error', 'Unknown error')}", "error")
            
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
            self.ui.add_response(format_error_message(f"Unknown command: '{cmd_text.strip()}'"), "error")
    
    def _process_command(self, command, cmd_text: str):
        """Process a parsed command using handler registry with fallback."""
        command_type = command.__class__.__name__
        
        # Try new handler system first
        handler = self.command_registry.get_handler(command_type)
        if handler:
            try:
                self.logger.debug(f"Using new handler for command: {command_type}")
                result = handler.handle(command, self)
                if not result.success:
                    self.ui.add_response(format_error_message(result.message), "error")
                return
            except Exception as e:
                self.logger.error(f"Handler failed for {command_type}: {str(e)}")
                self.ui.add_response(format_error_message(f"Command handler failed: {str(e)}"), "error")
                return
        
        # All commands have been migrated to handlers
        self.logger.warning(f"No handler found for command type: {command_type}")
        self.ui.add_response(format_error_message(f"No handler available for command: {command_type}"), "error")
    
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

            # If we reach here, the vector store loaded successfully (or would have thrown an exception)
            self.ui.add_response(f"\n{Emojis.CHECK} Vector database loaded successfully from:"
                                 f" {self.rag_service.db_path}", "info")
            if hasattr(self.rag_service, 'db_info') and self.rag_service.db_info:
                self.ui.add_response(f"\n{Emojis.INFO} Auto-detected embedding model: "
                                     f"{self.rag_service.embedding_model}", "info")
        except Exception as e:
            print(f"Error during UI startup: {str(e)}")
            # If the UI failed to create properly, handle gracefully
            if not ui_created:
                print(banner)
                print("\nError initializing UI. Check your terminal configuration.")
                return
            
        # Display success and model info
        self.ui.add_response(f"\n{Emojis.ROBOT} Using LLM model: {self.rag_service.llm_model}", "info")
        self.ui.add_response(f"\n{Emojis.CHUNKS} Number of chunks to retrieve: "
                             f"{self.rag_service.num_chunks}", "info")
        self.ui.add_response(f"\n{Emojis.INFO} Note: The first query may take longer as the "
                             f"model needs to load.", "info")

        # Run the UI
        try:
            self.ui.run()
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
        finally:
            # Clean shutdown - wait for LLM threads to complete
            self.shutdown()
            
            # Handle auto-save after shutdown
            if auto_save and self.rag_service.conversation_history:
                try:
                    result = self.rag_service.save_history()
                    if result['success']:
                        print(f"\nConversation history saved to {result['filepath']}")
                    else:
                        print(f"\nError saving conversation history: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"\nError saving conversation history: {str(e)}")