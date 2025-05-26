"""
Command completion functionality for the Ragger command mode.
"""
from prompt_toolkit.completion import Completer, Completion


class CommandCompleter(Completer):
    """Completer for Ragger commands with descriptions."""
    
    def __init__(self):
        """Initialize the command completer with available commands."""
        # Set max completions to show (can generate up to 10 completions)
        self.max_completions = 10
        
        # Full commands with descriptions
        self.command_list = [
            # Essential commands
            ("exit", "Exit the application"),
            ("quit", "Exit the application"),
            ("q", "Exit the application"),
            ("help", "Show help information"),
            ("h", "Show help information"),
            
            # History command
            ("history", "Show conversation history"),
            
            # Clear commands
            ("clear", "Clear all history and custom chunks"),
            ("clear-history", "Clear only conversation history"),
            ("ch", "Clear only conversation history"),
            ("clear-chunks", "Clear only custom chunks"),
            ("cc", "Clear only custom chunks"),
            
            # Save command
            ("save", "Save conversation history"),
            
            # Chunk management
            ("add", "Add chunk to custom chunks"),
            ("a", "Add chunk to custom chunks (shortcut)"),
            ("list-chunks", "List all custom chunks"),
            ("lc", "List all custom chunks (shortcut)"),
            ("expand", "Expand a chunk from search results"),
            ("e", "Expand a chunk from search results (shortcut)"),
            ("expand-custom", "Expand a chunk from custom chunks"),
            ("ec", "Expand a chunk from custom chunks (shortcut)"),
            ("set-chunks", "Set number of chunks to retrieve"),
            ("sc", "Set number of chunks to retrieve (shortcut)"),
            ("get-chunks", "Display current chunks setting"),
            ("gc", "Display current chunks setting (shortcut)"),
            
            # Prompt commands
            ("add-prompt", "Add chunk to prompt"),
            ("ap", "Add chunk to prompt (shortcut)"),
            ("prompt", "Send a prompt directly"),
            ("p", "Send a prompt directly (shortcut)"),
            
            # Search commands
            ("search", "Search vector database"),
            ("s", "Search vector database (shortcut)")
        ]
        
        # List of just the command names for easy lookup
        self.commands = [cmd for cmd, _ in self.command_list]
    
    def get_completions(self, document, complete_event):
        """Get command completions based on the user's input.
        
        Args:
            document: The Document instance for the current input
            complete_event: The CompleteEvent that triggered this completion
            
        Yields:
            Completion instances for matching commands with descriptions
        """
        text = document.text.lstrip()
        
        # If there's a space, don't complete (only complete commands, not arguments)
        if ' ' in text:
            return
            
        # Find matching commands and yield them with descriptions
        for command, description in self.command_list:
            if command.startswith(text):
                # Calculate the start position (length of the document minus what has been typed)
                start_position = -len(text)
                
                # Yield the completion with metadata (description)
                yield Completion(
                    command, 
                    start_position=start_position,
                    display_meta=description
                )