"""
Terminal UI using prompt_toolkit.

This module provides the RaggerUI class which implements a terminal
user interface using prompt_toolkit.
"""
from typing import Callable

from prompt_toolkit import Application
from prompt_toolkit.styles import Style
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import DynamicCompleter
from prompt_toolkit.filters import has_focus, Condition
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.layout import Layout, Window, HSplit, VSplit, FormattedTextControl, BufferControl

from ragger.ui.terminal.mode import InputMode
from ragger.ui.terminal.app_state import AppState
from ragger.ui.resources import Emojis, UIColors, HELP_TEXT

from ragger.core.completer import CommandCompleter


class RaggerUI:
    """Prompt-toolkit based UI for the Ragger application."""

    def __init__(self, on_query: Callable, on_command: Callable, debug_mode: bool = False):
        """Initialize the UI components.
        
        Args:
            on_query: Callback for handling queries
            on_command: Callback for handling commands
            debug_mode: Enable debug messages for easier troubleshooting
        """
        # Initialize application state
        self.state = AppState(on_query, on_command)

        # Debug mode for showing internal errors
        self.debug_mode = debug_mode

        # Collect debug messages for post-exit review
        # Instead of showing in UI which causes scrolling issues
        self.debug_log = []

        # Create UI styles
        self.style = Style.from_dict({
            # General UI elements
            'header': f'bg:{UIColors.HEADER_BG} {UIColors.HEADER_FG} bold',
            'footer': f'bg:{UIColors.HEADER_BG} {UIColors.HEADER_FG}',
            'output-area': f'bg:{UIColors.OUTPUT_AREA_BG} {UIColors.OUTPUT_AREA_FG}',
            'input-area': f'bg:{UIColors.INPUT_AREA_BG} {UIColors.INPUT_AREA_FG}',
            'status-bar': f'bg:{UIColors.STATUS_BAR_BG} {UIColors.STATUS_BAR_FG}',
            'status-bar bold': f'bg:{UIColors.STATUS_BAR_BG} {UIColors.STATUS_BAR_FG} bold',

            # Mode indicators (foreground only, no background)
            'mode.prompt': f'{UIColors.PROMPT_MODE_FG} bold',
            'mode.command': f'{UIColors.COMMAND_MODE_FG} bold',
            'mode.search': f'{UIColors.SEARCH_MODE_FG} bold',

            # Message categories
            'message.info': f'{UIColors.INFO_FG}',
            'message.prompt': f'{UIColors.PROMPT_FG} bold',
            'message.prompt_header': f'{UIColors.PROMPT_FG} bold',
            'message.response': f'{UIColors.RESPONSE_FG}',
            'message.response_header': f'{UIColors.RESPONSE_FG} bold',
            'message.error': f'{UIColors.ERROR_FG} bold',
            'message.warning': f'{UIColors.WARNING_FG} bold',
            'message.system': f'{UIColors.SYSTEM_FG} italic',
            'message.chunks_header': f'{UIColors.INFO_FG} bold',
            'message.chunk': f'{UIColors.INFO_FG}',

            # Scrollbar
            'scrollbar.background': f'bg:{UIColors.SCROLLBAR_BG}',
            'scrollbar.button': f'bg:{UIColors.SCROLLBAR_BUTTON}',

            # Completion menu styling
            'completion-menu': 'bg:#2a2a2a #ffffff',
            'completion-menu.completion': 'bg:#333333 #ffffff',
            'completion-menu.completion.current': 'bg:#004488 #ffffff',
            'completion-menu.meta.completion': 'bg:#333333 #888888',
            'completion-menu.meta.completion.current': 'bg:#004488 #cccccc',
            'completion-menu.border': 'bg:#2a2a2a #ffffff',
        })

        # Create buffers for input and output
        self.output_buffer = Buffer(read_only=True)

        # Create command completer
        self.command_completer = CommandCompleter()

        # Function to conditionally use the completer
        @Condition
        def is_command_mode():
            return self.state.current_mode == InputMode.COMMAND

        # Set completer max completions
        self.command_completer.max_completions = 10

        # Create input buffer with completer
        self.input_buffer = Buffer(
            multiline=True,
            auto_suggest=AutoSuggestFromHistory(),
            completer=DynamicCompleter(lambda: self.command_completer if is_command_mode() else None),
            complete_while_typing=True
        )

        # Create a new KeyBindings instance
        self.kb = KeyBindings()

        @self.kb.add('escape', filter=has_focus(self.input_buffer))
        def _(event):
            buffer = event.app.current_buffer
            if buffer.complete_state:
                buffer.cancel_completion()

        # Tab key binding for mode switching
        @self.kb.add('tab')
        def _(event):
            """Toggle between input modes with Tab."""
            # Save current buffer content before switching
            self.state.set_current_buffer_content(self.input_buffer.text)

            # Switch mode
            self.state.toggle_mode()

            # Update buffer with new mode's content
            self.input_buffer.text = self.state.get_current_buffer_content()
            self._update_ui()

        @self.kb.add('f2')
        def _(event):
            """Show help information."""
            self._show_help()

        # Handle Enter in input buffer
        @self.kb.add('enter', filter=has_focus(self.input_buffer))
        def _(event):
            """Handle Enter in input buffer."""
            # Get current text
            text = self.input_buffer.text

            # Skip if empty
            if not text.strip():
                return

            # Save to history
            self.state.get_current_history().append_string(text)

            # Handle based on mode
            match self.state.current_mode:
                case InputMode.PROMPT:
                    # Add to output history
                    self.state.add_to_output_history(f"User: {text}", "prompt")
                    # Process query
                    self.state.on_query(text)

                case InputMode.COMMAND:
                    # Add to output history
                    self.state.add_to_output_history(f"Command: {text}", "system")
                    # Process command
                    self.state.on_command(text)

                case InputMode.SEARCH:
                    # Add to output history
                    self.state.add_to_output_history(f"Search: {text}", "system")
                    # Process as search command
                    self.state.on_command(f"search {text}")

            # Clear input buffer
            self.input_buffer.text = ""

            # Reset history index
            self.state.set_current_history_index(0)

            # Update UI
            self._update_output_buffer()

        # Insert newline in any mode (all modes support multiline input)
        @self.kb.add('c-j', filter=has_focus(self.input_buffer))
        def _(event):
            """Insert newline in any mode."""
            self.input_buffer.insert_text('\n')

        self.bindings = self.kb

        # Add key handlers for basic navigation and editing

        @self.kb.add('backspace', filter=has_focus(self.input_buffer))
        def _(event):
            """Handle backspace in input buffer."""
            # Delete the character before the cursor
            self.input_buffer.delete_before_cursor(1)

        @self.kb.add('delete', filter=has_focus(self.input_buffer))
        def _(event):
            """Handle delete in input buffer."""
            # Delete the character at the cursor
            self.input_buffer.delete(1)

        @self.kb.add('left', filter=has_focus(self.input_buffer))
        def _(event):
            """Move cursor left in input buffer or cycle through completions."""
            buffer = event.app.current_buffer

            # If command mode and completion menu is showing, select previous
            if self.state.current_mode == InputMode.COMMAND and buffer.complete_state:
                buffer.complete_previous()
            else:
                # Otherwise just move cursor left
                buffer.cursor_left()

        @self.kb.add('right', filter=has_focus(self.input_buffer))
        def _(event):
            """Move cursor right in input buffer or cycle through completions."""
            buffer = event.app.current_buffer

            # If command mode and completion menu is showing, select next
            if self.state.current_mode == InputMode.COMMAND and buffer.complete_state:
                buffer.complete_next()
            else:
                # Otherwise just move cursor right
                buffer.cursor_right()

        @self.kb.add('up', filter=has_focus(self.input_buffer))
        def _(event):
            """Navigate up through history or move cursor up if multiline."""

            # If we're in a multiline input and not on the first line, move cursor up
            if '\n' in self.input_buffer.text \
                    and hasattr(self.input_buffer, 'document') \
                    and self.input_buffer.document.cursor_position_row > 0:

                self.input_buffer.cursor_up()

            # Otherwise navigate through history
            else:
                # Save current text if we're at the beginning of history navigation
                if self.state.get_current_history_index() == 0:
                    current_text = self.input_buffer.text
                    if current_text:
                        self.state.set_current_buffer_content(current_text)

                # Navigate to previous item (older entry)
                previous_item = self.state.get_history_item(-1)
                if previous_item:
                    self.input_buffer.text = previous_item

        @self.kb.add('down', filter=has_focus(self.input_buffer))
        def _(event):
            """Navigate down through history or move cursor down if multiline."""
            # If we're in a multiline input and not on the last line, move cursor down
            if '\n' in self.input_buffer.text and hasattr(self.input_buffer, 'document') \
                    and self.input_buffer.document.cursor_position_row < self.input_buffer.document.line_count - 1:
                self.input_buffer.cursor_down()
            else:
                # Otherwise navigate through history
                # Navigate to next item (newer entry)
                next_item = self.state.get_history_item(1)
                if next_item:
                    self.input_buffer.text = next_item
                else:
                    # If at the newest entry, restore the saved content
                    self.input_buffer.text = self.state.get_current_buffer_content()

        # Scroll step sizes
        self.scroll_step_small = 5
        self.scroll_step_large = 40

        @self.kb.add('s-up')
        def _(event):
            """Scroll the history upward (toward the top/older content)."""
            for _ in range(self.scroll_step_small):
                if not self._scroll_up():
                    break

        @self.kb.add('s-down')
        def _(event):
            """Scroll the history downward."""
            for _ in range(self.scroll_step_small):
                if not self._scroll_down():
                    break

        @self.kb.add('pageup')
        def _(event):
            """Fast scroll up with Page Up."""
            for _ in range(self.scroll_step_large):
                if not self._scroll_up():
                    break
                event.app.invalidate()

        @self.kb.add('pagedown')
        def _(event):
            """Very fast scroll down with Page Down."""
            for _ in range(self.scroll_step_large):
                if not self._scroll_down():
                    break
                event.app.invalidate()

        @self.kb.add('home')
        def _(event):
            """Jump max 10000 lines up.
             (avoid infinite loop instead of while)
             TODO: Refactor
             """
            for _ in range(10000):
                start = self._scroll_up()
                event.app.invalidate()
                if not start:
                    break

        @self.kb.add('end')
        def _(event):
            """Jump max 10000 lines down to the end """
            for _ in range(10000):
                end = self._scroll_down()
                event.app.invalidate()
                if not end:
                    break

        @self.kb.add('c-c')
        def _(event):
            """Exit application on Ctrl+C."""
            event.app.exit()

        # Create layout elements
        header = Window(
            content=FormattedTextControl(lambda: [('class:header', ' RAGGER - RAG Query Interface ')]),
            height=1,
            style='class:header',
        )

        # Create a simple output control that is not focusable
        output_control = BufferControl(
            buffer=self.output_buffer,
            focusable=False,
            focus_on_click=False,
        )

        # Create output area with scrolling and selection support
        self.output_window = Window(
            content=output_control,
            style='class:output-area',
            wrap_lines=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
            allow_scroll_beyond_bottom=True,
            dont_extend_height=False,
        )

        # Create status bar
        status_bar = Window(
            content=FormattedTextControl(self._get_status_bar_text),
            height=1,
            style='class:status-bar',
        )

        # Create input area with mode indicator
        mode_indicator = Window(
            content=FormattedTextControl(self._get_mode_indicator_text),
            width=3,  # Just enough for the emoji and spacing
            style='class:mode-indicator',
        )

        # Create input window (smaller height to give more space to output)
        self.input_window = Window(
            content=BufferControl(
                buffer=self.input_buffer,
                # No menu_position as we're using CompletionsMenu container
            ),
            height=3,
            style='class:input-area',
            wrap_lines=True,
            dont_extend_height=False,  # Allow extension for completions
            allow_scroll_beyond_bottom=True
        )

        # Create input container with mode indicator and input window
        input_container = VSplit([
            mode_indicator,
            self.input_window,
        ])

        # Create completion menu with max 5 visible suggestions
        completion_menu = CompletionsMenu(max_height=5, scroll_offset=0)

        # Create root container with all UI elements
        root_container = HSplit([
            header,
            self.output_window,
            status_bar,
            input_container,
            completion_menu
        ])

        # Create layout with focus on input window
        self.layout = Layout(root_container, focused_element=self.input_window)

        # Create the application
        self.app = Application(
            layout=self.layout,
            key_bindings=self.bindings,
            style=self.style,
            mouse_support=True,
            full_screen=True,
            min_redraw_interval=0.05
        )

    def _scroll_up(self) -> bool:
        """Scroll the output window up one position

        Returns:
            bool: True if scrolled, False otherwise
        """
        if not (info := self.output_window.render_info):
            return False

        if self.output_window.vertical_scroll > 0:
            # From Prompt-Toolkit: not entirely correct yet in case of line wrapping and long lines.
            if (
                    info.cursor_position.y >=
                    info.window_height - 1 - info.configured_scroll_offsets.bottom
            ):
                self.output_window.content.move_cursor_up()

            self.output_window.vertical_scroll -= 1
            return True

        return False

    def _scroll_down(self) -> bool:
        """Scroll the output window down one position

        Returns:
            bool: True if scrolled, False otherwise
        """
        if not (info := self.output_window.render_info):
            return False
        if self.output_window.vertical_scroll < info.content_height - info.window_height:
            if info.cursor_position.y <= info.configured_scroll_offsets.top:
                self.output_window.content.move_cursor_down()

            self.output_window.vertical_scroll += 1
            return True
        return False

    def _get_mode_indicator_text(self):
        """Get formatted text for the mode indicator."""
        match self.state.current_mode:
            case InputMode.PROMPT:
                return [('class:mode.prompt', f'{Emojis.PROMPT} ')]
            case InputMode.COMMAND:
                return [('class:mode.command', f'{Emojis.COMMAND} ')]
            case InputMode.SEARCH:
                return [('class:mode.search', f'{Emojis.SEARCH} ')]
            case _:
                return [('', '')]

    def _get_status_bar_text(self):
        """Get formatted text for the status bar."""
        # Get mode-specific instruction text
        match self.state.current_mode:
            case InputMode.PROMPT:
                instruction = "Enter query"
                additional_help = "Ctrl+J: Newline"
            case InputMode.COMMAND:
                instruction = "Enter command"
                additional_help = ""
            case InputMode.SEARCH:
                instruction = "Enter search terms"
                additional_help = ""
            case _:
                instruction = ""
                additional_help = ""

        # Build status text
        status_parts = [
            ('class:status-bar', ' '),
            ('class:status-bar bold', f'{instruction}'),  # Bold instruction
            ('class:status-bar', ' | '),
            ('class:status-bar.chunks',
             f'Custom: {self.state.custom_chunks_count} | Retrieved: {self.state.retrieved_chunks_count} '),
            ('class:status-bar', ' | ')
        ]

        # Only add additional help if available
        if additional_help:
            status_parts.append(('class:status-bar.keys', f'{additional_help} | '))

        # Always add main shortcuts
        status_parts.append(('class:status-bar.keys', f'Tab: Switch Mode | F2: Help | Ctrl+C: Exit'))

        return status_parts

    def _update_output_buffer(self):
        """Update the output buffer with current history."""
        # Format all history items
        output_text = ""
        for text, category in self.state.output_history:
            # Add plain text without formatting tags
            output_text += f"{text}\n"

        # Set output buffer text (with bypass for read-only)
        self.output_buffer.set_document(
            Document(output_text),
            bypass_readonly=True
        )

        # Force a UI update
        self.app.invalidate()

    def _update_ui(self):
        """Update various UI components."""
        # Update any dynamic UI elements
        self.app.invalidate()

    def add_response(self, text: str, category: str = "response"):
        """Add a response to the output history."""
        self.state.add_to_output_history(text, category)
        self._update_output_buffer()

        # Force immediate UI update to ensure chunks appear before LLM generation
        self.app.invalidate()
        
        # Force the output buffer to scroll to the end to show new content
        if hasattr(self.output_window, 'content') and hasattr(self.output_window.content, 'buffer'):
            # Move cursor to end of buffer to ensure new content is visible
            buffer = self.output_window.content.buffer
            buffer.cursor_position = len(buffer.text)
            
        # Schedule an immediate redraw by calling output processing
        if hasattr(self.app, 'output') and hasattr(self.app.output, 'flush'):
            try:
                self.app.output.flush()
            except:
                pass

    def set_chunk_counts(self, custom_count: int, retrieved_count: int):
        """Set the chunk counts for the status bar."""
        self.state.custom_chunks_count = custom_count
        self.state.retrieved_chunks_count = retrieved_count
        self._update_ui()

    def _log_debug(self, message: str, category: str = "debug"):
        """Add a debug message to the debug log if debug mode is enabled.
        
        Args:
            message: Debug message to log
            category: Category for styling (debug, error, warning, etc.)
        """
        if self.debug_mode:
            # Get current timestamp for debugging precision
            from datetime import datetime
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # Add to debug log with timestamp and category, but NOT to the UI buffer
            # This ensures debug messages don't affect scrolling behavior
            self.debug_log.append(f"[{timestamp}] {category.upper()}: {message}")

    def _show_help(self):
        """Display help information in the output buffer."""
        # Add help text to output history
        self.state.add_to_output_history(HELP_TEXT, "info")
        self._update_output_buffer()

    def run(self):
        """Run the UI application."""
        # Initial UI update
        self._update_output_buffer()

        try:
            # Run the application
            self.app.run()
        finally:
            # Print debug log after the UI closes
            if self.debug_mode and self.debug_log:
                print("\n=== DEBUG LOG ===")
                for i, entry in enumerate(self.debug_log, 1):
                    print(f"{i}. {entry}")
                print("==================")
