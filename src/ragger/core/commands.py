"""
Command management and processing for the Ragger application.

This module handles command recognition, parsing, and execution
independent of the user interface.
"""
import re
from typing import Optional, Tuple, Dict, Any, List


class Command:
    """Base class for all commands."""
    
    def __init__(self, text: str, args: List[str]):
        self.text = text
        self.args = args
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.args})"


class AddCommand(Command):
    """Command to add chunk content to the prompt."""
    
    def __init__(self, text: str, args: List[str]):
        super().__init__(text, args)
        
        # Default values
        self.chunk_num = None
        self.context_size = 500
        self.position = "start"
        
        # Parse arguments
        if len(args) >= 2:
            try:
                self.chunk_num = int(args[1])
            except (ValueError, IndexError):
                pass
                
        # Optional context size
        if len(args) >= 3:
            try:
                self.context_size = int(args[2])
            except (ValueError, IndexError):
                pass
                
        # Position (start or end)
        for i in range(2, min(len(args), 4)):
            if args[i].lower() == "end":
                self.position = "end"
                break
            elif args[i].lower() == "start":
                self.position = "start"
                break


class ExpandCommand(Command):
    """Command to expand a chunk's context."""
    
    def __init__(self, text: str, args: List[str]):
        super().__init__(text, args)
        
        # Default values
        self.chunk_num = None
        self.context_size = 500
        
        # Parse arguments
        if len(args) >= 2:
            try:
                self.chunk_num = int(args[1])
            except (ValueError, IndexError):
                pass
                
        # Optional context size
        if len(args) >= 3:
            try:
                self.context_size = int(args[2])
            except (ValueError, IndexError):
                pass


class HistoryCommand(Command):
    """Command to show conversation history."""
    pass


class ClearCommand(Command):
    """Command to clear all history and chunks."""
    pass


class ClearHistoryCommand(Command):
    """Command to clear only conversation history."""
    pass


class ClearChunksCommand(Command):
    """Command to clear only custom chunks."""
    pass


class SaveCommand(Command):
    """Command to save conversation history."""
    pass


class ExitCommand(Command):
    """Command to exit the application."""
    pass


class ToggleModeCommand(Command):
    """Command to toggle between prompt and command modes."""
    pass


class HistoryUpCommand(Command):
    """Command to navigate up in history."""
    pass


class HistoryDownCommand(Command):
    """Command to navigate down in history."""
    pass


class SetChunksCommand(Command):
    """Command to set the number of chunks to retrieve."""
    
    def __init__(self, text: str, args: List[str]):
        super().__init__(text, args)
        
        # Default values
        self.num_chunks = None
        
        # Parse arguments
        if len(args) >= 2:
            try:
                self.num_chunks = int(args[1])
            except (ValueError, IndexError):
                pass
                

class GetChunksCommand(Command):
    """Command to get the current number of chunks to retrieve."""
    pass


class ListChunksCommand(Command):
    """Command to list custom chunks."""
    pass


class AddPromptCommand(Command):
    """Command to add a chunk from custom chunks to prompt."""
    
    def __init__(self, text: str, args: List[str]):
        super().__init__(text, args)
        
        # Default values
        self.chunk_num = None
        self.context_size = 500
        self.position = "start"
        
        # Parse arguments
        if len(args) >= 2:
            try:
                self.chunk_num = int(args[1])
            except (ValueError, IndexError):
                pass
                
        # Optional context size
        if len(args) >= 3:
            try:
                self.context_size = int(args[2])
            except (ValueError, IndexError):
                pass
                
        # Position (start or end)
        for i in range(2, min(len(args), 4)):
            if args[i].lower() == "end":
                self.position = "end"
                break
            elif args[i].lower() == "start":
                self.position = "start"
                break


class SearchCommand(Command):
    """Command to search the vector store for chunks."""
    
    def __init__(self, text: str, args: List[str]):
        super().__init__(text, args)
        
        # Default values
        self.query = None
        self.num_chunks = None
        
        # If args contain more than the command itself, the rest is the query
        if len(args) >= 2:
            self.query = " ".join(args[1:])
            
        # Check if last argument is a number prefixed with #
        if len(args) >= 2 and args[-1].startswith('#'):
            try:
                self.num_chunks = int(args[-1][1:])
                # Remove the number from the query
                if len(args) > 2:
                    self.query = " ".join(args[1:-1])
                else:
                    self.query = None
            except (ValueError, IndexError):
                pass


class PromptCommand(Command):
    """Command to send a prompt directly to the LLM."""
    
    def __init__(self, text: str, args: List[str]):
        super().__init__(text, args)
        
        # Default values
        self.query = None
        
        # If args contain more than the command itself, the rest is the query
        if len(args) >= 2:
            self.query = " ".join(args[1:])
            


class CommandResult:
    """Result of a command execution."""
    
    def __init__(self, success: bool, message: str, data: Dict[str, Any] = None):
        self.success = success
        self.message = message
        self.data = data or {}


class ContentProcessor:
    """Process and clean content from command patterns."""
    
    def __init__(self):
        # Command patterns for full line matching
        self.cmd_patterns = [
            r'^a\s+\d+(\s+\d+)?(\s+(start|end))?$',    # a N [size] [position]
            r'^add\s+\d+(\s+\d+)?(\s+(start|end))?$',  # add N [size] [position]
            r'^e\s+\d+(\s+\d+)?$',                     # e N [size]
            r'^expand\s+\d+(\s+\d+)?$',                # expand N [size]
            r'^a$',                                    # just a
            r'^add$',                                  # just add
            r'^e$',                                    # just e
            r'^expand$',                               # just expand
            r'^ap\s+\d+(\s+\d+)?(\s+(start|end))?$',   # ap N [size] [position]
            r'^add-prompt\s+\d+(\s+\d+)?(\s+(start|end))?$', # add-prompt N [size] [position]
            r'^lc$',                                   # list chunks
            r'^list-chunks$',                          # list chunks
            r'^ch$',                                   # clear history
            r'^clear-history$',                        # clear history
            r'^cc$',                                   # clear chunks
            r'^clear-chunks$',                         # clear chunks
            r'^s(\s+.+)?$',                            # search [query]
            r'^search(\s+.+)?$',                       # search [query]
            r'^p(\s+.+)?$',                            # prompt [query]
            r'^prompt(\s+.+)?$',                       # prompt [query]
            r'^sc\s+\d+$',                             # set-chunks N
            r'^set-chunks\s+\d+$',                     # set-chunks N
            r'^gc$',                                   # get-chunks
            r'^get-chunks$',                           # get-chunks
            r'^mode$',                                 # toggle mode
            r'^tab$',                                  # toggle mode (alternative)
            r'^up$',                                   # history up
            r'^down$',                                 # history down
        ]
    
    def clean_commands(self, text: str) -> str:
        """Remove command patterns from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text with command patterns removed
        """
        if not text:
            return text
            
        # Preserve original line endings
        ends_with_newline = text.endswith('\n')
        
        # Split into lines for processing
        lines = text.split('\n')
        filtered_lines = []
        
        # Process each line
        for line in lines:
            line_stripped = line.strip()
            
            # Check if the line is just a command
            if any(re.match(pattern, line_stripped) for pattern in self.cmd_patterns):
                continue  # Skip command lines
            else:
                filtered_lines.append(line)
        
        # Reconstruct with original ending
        result = '\n'.join(filtered_lines)
        if ends_with_newline and result:
            result += '\n'
            
        return result
    
    def clean_trailing_command(self, text: str) -> str:
        """Clean only trailing commands, preserving mixed content.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with trailing commands removed
        """
        if not text:
            return text
            
        # Use extract_trailing_command to do the heavy lifting
        content, _ = self.extract_trailing_command(text)
        return content
    
    def extract_trailing_command(self, text: str) -> Tuple[str, Optional[str]]:
        """Extract trailing command from text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (content, command) where command may be None
        """
        if not text:
            return text, None
            
        # Split into lines
        lines = text.split('\n')
        command = None
        
        # Check the last non-empty line for a command
        for i in range(len(lines)-1, -1, -1):
            if not lines[i].strip():
                continue  # Skip empty lines
                
            line_stripped = lines[i].strip()
            if any(re.match(pattern, line_stripped) for pattern in self.cmd_patterns):
                command = line_stripped
                lines[i] = ""  # Remove the command line
                break  # Found a command, stop looking
            else:
                break  # Found non-empty content that's not a command, stop looking
            
        # Remove trailing empty lines
        while lines and not lines[-1].strip():
            lines.pop()
            
        # Reconstruct content
        content = '\n'.join(lines)
        if text.endswith('\n') and content:
            content += '\n'
            
        return content, command


class CommandManager:
    """Handles command parsing, recognition and execution."""
    
    def __init__(self):
        """Initialize the command manager."""
        self.content_processor = ContentProcessor()
        
        # Command pattern mapping
        self.command_patterns = {
            r'^(a|add)$': self._create_add_command,
            r'^(a|add)\s+\d+': self._create_add_command,
            r'^(e|expand)\s+\d+': self._create_expand_command,
            r'^history$': self._create_history_command,
            r'^clear$': self._create_clear_command,
            r'^(ch|clear-history)$': self._create_clear_history_command,
            r'^(cc|clear-chunks)$': self._create_clear_chunks_command,
            r'^(lc|list-chunks)$': self._create_list_chunks_command,
            r'^(ap|add-prompt)\s+\d+': self._create_add_prompt_command,
            r'^(s|search)(\s+.+)?$': self._create_search_command,
            r'^(p|prompt)(\s+.+)?$': self._create_prompt_command,
            r'^save$': self._create_save_command,
            r'^(exit|quit|q)$': self._create_exit_command,
            r'^mode$': self._create_toggle_mode_command,
            r'^tab$': self._create_toggle_mode_command,
            r'^up$': self._create_history_up_command,
            r'^down$': self._create_history_down_command,
            r'^(sc|set-chunks)\s+\d+': self._create_set_chunks_command,
            r'^(gc|get-chunks)$': self._create_get_chunks_command,
        }
    
    def parse_input(self, text: str) -> Optional[Command]:
        """Parse input to determine if it's a command.
        
        Args:
            text: Input text
            
        Returns:
            Command object if input is a command, None otherwise
        """
        if not text:
            return None
            
        # Clean and normalize input
        text_lower = text.lower().strip()
        
        # Check against command patterns
        for pattern, create_func in self.command_patterns.items():
            if re.match(pattern, text_lower):
                args = text_lower.split()
                return create_func(text, args)
                
        return None
    
    def is_pure_command(self, text: str) -> bool:
        """Check if text is purely a command.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is a command, False otherwise
        """
        return self.parse_input(text) is not None
    
    def extract_trailing_command(self, text: str) -> Tuple[str, Optional[Command]]:
        """Extract trailing command from text if present.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (content, command) where command may be None
        """
        content, cmd_text = self.content_processor.extract_trailing_command(text)
        
        if cmd_text:
            cmd = self.parse_input(cmd_text)
            return content, cmd
            
        return text, None
    
    def clean_commands(self, text: str) -> str:
        """Clean command patterns from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return self.content_processor.clean_commands(text)
    
    def clean_trailing_command(self, text: str) -> str:
        """Clean trailing command from text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        return self.content_processor.clean_trailing_command(text)
    
    # Command creation methods
    def _create_add_command(self, text: str, args: List[str]) -> AddCommand:
        return AddCommand(text, args)
        
    def _create_expand_command(self, text: str, args: List[str]) -> ExpandCommand:
        return ExpandCommand(text, args)
        
    def _create_history_command(self, text: str, args: List[str]) -> HistoryCommand:
        return HistoryCommand(text, args)
        
    def _create_clear_command(self, text: str, args: List[str]) -> ClearCommand:
        return ClearCommand(text, args)
        
    def _create_clear_history_command(self, text: str, args: List[str]) -> ClearHistoryCommand:
        return ClearHistoryCommand(text, args)
        
    def _create_clear_chunks_command(self, text: str, args: List[str]) -> ClearChunksCommand:
        return ClearChunksCommand(text, args)
        
    def _create_list_chunks_command(self, text: str, args: List[str]) -> ListChunksCommand:
        return ListChunksCommand(text, args)
        
    def _create_add_prompt_command(self, text: str, args: List[str]) -> AddPromptCommand:
        return AddPromptCommand(text, args)
        
    def _create_search_command(self, text: str, args: List[str]) -> SearchCommand:
        return SearchCommand(text, args)
        
    def _create_save_command(self, text: str, args: List[str]) -> SaveCommand:
        return SaveCommand(text, args)
        
    def _create_exit_command(self, text: str, args: List[str]) -> ExitCommand:
        return ExitCommand(text, args)
        
    def _create_toggle_mode_command(self, text: str, args: List[str]) -> ToggleModeCommand:
        return ToggleModeCommand(text, args)
        
    def _create_history_up_command(self, text: str, args: List[str]) -> HistoryUpCommand:
        return HistoryUpCommand(text, args)
        
    def _create_history_down_command(self, text: str, args: List[str]) -> HistoryDownCommand:
        return HistoryDownCommand(text, args)
        
    def _create_set_chunks_command(self, text: str, args: List[str]) -> SetChunksCommand:
        return SetChunksCommand(text, args)
        
    def _create_get_chunks_command(self, text: str, args: List[str]) -> GetChunksCommand:
        return GetChunksCommand(text, args)
        
    def _create_prompt_command(self, text: str, args: List[str]) -> PromptCommand:
        return PromptCommand(text, args)