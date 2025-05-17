"""
Application state management for the terminal UI.

This module provides the AppState class to manage the state of the
terminal UI, including mode, history, and content buffers.
"""
from typing import Callable, List, Tuple

from prompt_toolkit.history import InMemoryHistory

from ragger.ui.terminal.mode import InputMode


class AppState:
    """Centralized state management for the terminal application."""
    
    def __init__(self, on_query: Callable, on_command: Callable):
        # Mode tracking
        self.current_mode = InputMode.PROMPT
        
        # Content buffers
        self.prompt_content = ""
        self.command_content = ""
        self.search_content = ""
        
        # History
        self.prompt_history = InMemoryHistory()
        self.command_history = InMemoryHistory()
        self.search_history = InMemoryHistory()
        
        # History navigation state for each mode
        self.prompt_history_index = 0
        self.command_history_index = 0
        self.search_history_index = 0
        
        # Output history
        self.output_history: List[Tuple[str, str]] = []
        
        # Statistics and info
        self.custom_chunks_count = 0
        self.retrieved_chunks_count = 0
        
        # Callbacks
        self.on_query = on_query
        self.on_command = on_command
        
        # Last mode for toggle logic
        self.last_mode = InputMode.PROMPT
        
    def toggle_mode(self):
        """Cycle through input modes with command mode as a hub.
        
        The mode cycling always goes through command mode:
        - From prompt mode → command mode
        - From search mode → command mode
        - From command mode → alternates between prompt and search
        """
        match self.current_mode:
            case InputMode.PROMPT:
                # From prompt always go to command
                self.current_mode = InputMode.COMMAND
                self.last_mode = InputMode.PROMPT
            case InputMode.SEARCH:
                # From search always go to command
                self.current_mode = InputMode.COMMAND
                self.last_mode = InputMode.SEARCH
            case InputMode.COMMAND:
                # From command, go to the mode we haven't visited recently
                if self.last_mode == InputMode.PROMPT:
                    self.current_mode = InputMode.SEARCH
                else:
                    self.current_mode = InputMode.PROMPT
                self.last_mode = InputMode.COMMAND
        
        # Reset history index when switching modes
        self.set_current_history_index(0)
        
    def get_current_buffer_content(self):
        """Get the content for the current mode."""
        match self.current_mode:
            case InputMode.PROMPT:
                return self.prompt_content
            case InputMode.COMMAND:
                return self.command_content
            case InputMode.SEARCH:
                return self.search_content
            case _:
                return ""
    
    def set_current_buffer_content(self, content: str):
        """Set the content for the current mode."""
        match self.current_mode:
            case InputMode.PROMPT:
                self.prompt_content = content
            case InputMode.COMMAND:
                self.command_content = content
            case InputMode.SEARCH:
                self.search_content = content
        
    def get_current_history(self):
        """Get the history for the current mode."""
        match self.current_mode:
            case InputMode.PROMPT:
                return self.prompt_history
            case InputMode.COMMAND:
                return self.command_history
            case InputMode.SEARCH:
                return self.search_history
            case _:
                return self.prompt_history
    
    def get_current_history_index(self):
        """Get the history index for the current mode."""
        match self.current_mode:
            case InputMode.PROMPT:
                return self.prompt_history_index
            case InputMode.COMMAND:
                return self.command_history_index
            case InputMode.SEARCH:
                return self.search_history_index
            case _:
                return 0
    
    def set_current_history_index(self, index):
        """Set the history index for the current mode."""
        match self.current_mode:
            case InputMode.PROMPT:
                self.prompt_history_index = index
            case InputMode.COMMAND:
                self.command_history_index = index
            case InputMode.SEARCH:
                self.search_history_index = index
    
    def get_history_item(self, offset):
        """Get a history item relative to the current index.
        
        Args:
            offset: Offset from current index (-1 for previous/older, +1 for next/newer)
            
        Returns:
            The history item or None if no item is available
        """
        history = self.get_current_history()
        
        # Get all history entries and reverse them (to get oldest first)
        entries = list(reversed(list(history.get_strings())))
        if not entries:
            return None
            
        # Get current index (0 = not in history yet, 1 = oldest item, etc.)
        current_index = self.get_current_history_index()
        
        if offset < 0:  # Up arrow - go back in history (to older items)
            new_index = current_index + abs(offset)  # Increase index
            
            # Make sure we don't go past the oldest entry
            if new_index > len(entries):
                new_index = len(entries)
                
            # If we're still in valid range
            if len(entries) >= new_index > 0:
                self.set_current_history_index(new_index)
                return entries[new_index - 1]  # -1 because index is 1-based
            return None
            
        else:  # Down arrow - go forward in history (to newer items)
            new_index = current_index - offset  # Decrease index
            
            # If we go past the newest entry, return None
            if new_index <= 0:
                self.set_current_history_index(0)  # Reset to not in history
                return None
                
            # If we're still in valid range
            if new_index <= len(entries):
                self.set_current_history_index(new_index)
                return entries[new_index - 1]  # -1 because index is 1-based
            return None
    
    def add_to_output_history(self, text: str, category: str = "info"):
        """Add text to the output history.
        
        Args:
            text: Text to add
            category: Category of text (info, error, prompt, response, etc.)
        """
        self.output_history.append((text, category))