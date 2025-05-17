"""
Copyright (c) 2025 Jakob Bolliger

This file is part of Ragger.

This program is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
The full text of the license can be found in the
LICENSE file in the root directory of this source tree.

UI resources for the Ragger application.

This module contains centralized UI elements like ASCII art, banners, colors,
and other visual resources used across different interfaces.
"""

# Emoji constants
class Emojis:
    """Class containing emoji constants to ensure consistent usage throughout the code."""
    CHECK = "✓"
    WARNING = "⚠️"
    ERROR = "❌"
    SEARCH = "🔍"
    RECEIVING = "📡"
    CHUNKS = "📚"
    CHUNK = "📌"
    USER = "👤"
    ROBOT = "🤖"
    WAITING = "⏳"
    BYE = "👋"
    CLEAN = "🧹"
    HISTORY = "📜"
    DETAILS = "🔍"
    BULLET = "•"
    HOURGLASS = "⌛"
    INFO = "ℹ️"
    PROMPT = "💬" 
    COMMAND = "🔧"
    UP = "⬆️"
    DOWN = "⬇️"
    TAB = "⇄"

# Border styles for box drawing
BORDER_STYLES = {
    "solid": {
        "top_left": "┌",
        "top_right": "┐",
        "bottom_left": "└",
        "bottom_right": "┘",
        "horizontal": "─",
        "vertical": "│"
    },
    "dashed": {
        "top_left": "┌",
        "top_right": "┐",
        "bottom_left": "└",
        "bottom_right": "┘",
        "horizontal": "┄",
        "vertical": "┆"
    }
}



# UI Color Configuration
class UIColors:
    """Color definitions for the UI components."""
    # General UI elements
    HEADER_BG = "#004488"
    HEADER_FG = "#ffffff"
    
    # Output and input areas now use terminal default colors (transparent)
    OUTPUT_AREA_BG = ""  # Empty for default/transparent
    OUTPUT_AREA_FG = ""  # Empty for default/transparent
    
    INPUT_AREA_BG = ""  # Empty for default/transparent
    INPUT_AREA_FG = ""  # Empty for default/transparent
    
    STATUS_BAR_BG = "#222222"
    STATUS_BAR_FG = "#eeeeee"
    
    # Mode indicator colors (foreground only, no background)
    PROMPT_MODE_FG = "#22dd22"
    COMMAND_MODE_FG = "#ddaa22"
    SEARCH_MODE_FG = "#2222dd"
    
    # Message categories
    INFO_FG = "#888888"
    PROMPT_FG = "#00aa00"
    RESPONSE_FG = "#0088dd"
    ERROR_FG = "#dd0000"
    WARNING_FG = "#ddaa00"
    SYSTEM_FG = "#aaaaaa"
    
    # Scrollbar
    SCROLLBAR_BG = "#222222"
    SCROLLBAR_BUTTON = "#444444"

# Main ASCII art banner used in both CLI and formatter
ASCII_BANNER = r"""
         _           _                   _              _              _            _
        /\ \        / /\                /\ \           /\ \           /\ \         /\ \
       /  \ \      / /  \              /  \ \         /  \ \         /  \ \       /  \ \
      / /\ \ \    / / /\ \            / /\ \_\       / /\ \_\       / /\ \ \     / /\ \ \
     / / /\ \_\  / / /\ \ \          / / /\/_/      / / /\/_/      / / /\ \_\   / / /\ \_\
    / / /_/ / / / / /  \ \ \        / / / ______   / / / ______   / /_/_ \/_/  / / /_/ / /
   / / /__\/ / / / /___/ /\ \      / / / /\_____\ / / / /\_____\ / /____/\    / / /__\/ /
  / / /_____/ / / /_____/ /\ \    / / /  \/____ // / /  \/____ // /\____\/   / / /_____/
 / / /\ \ \  / /_________/\ \ \  / / /_____/ / // / /_____/ / // / /______  / / /\ \ \
/ / /  \ \ \/ / /_       __\ \_\/ / /______\/ // / /______\/ // / /_______\/ / /  \ \ \
\/_/    \_\/\_\___\     /____/_/\/___________/ \/___________/ \/__________/\/_/    \_\/
"""

# Boxed version with border for modern CLI interface
BOXED_BANNER = r"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║          _           _                   _              _              _            _      ║
║         /\ \        / /\                /\ \           /\ \           /\ \         /\ \    ║
║        /  \ \      / /  \              /  \ \         /  \ \         /  \ \       /  \ \   ║
║       / /\ \ \    / / /\ \            / /\ \_\       / /\ \_\       / /\ \ \     / /\ \ \  ║
║      / / /\ \_\  / / /\ \ \          / / /\/_/      / / /\/_/      / / /\ \_\   / / /\ \_\ ║
║     / / /_/ / / / / /  \ \ \        / / / ______   / / / ______   / /_/_ \/_/  / / /_/ / / ║
║    / / /__\/ / / / /___/ /\ \      / / / /\_____\ / / / /\_____\ / /____/\    / / /__\/ /  ║
║   / / /_____/ / / /_____/ /\ \    / / /  \/____ // / /  \/____ // /\____\/   / / /_____/   ║
║  / / /\ \ \  / /_________/\ \ \  / / /_____/ / // / /_____/ / // / /______  / / /\ \ \     ║
║ / / /  \ \ \/ / /_       __\ \_\/ / /______\/ // / /______\/ // / /_______\/ / /  \ \ \    ║
║ \/_/    \_\/\_\___\     /____/_/\/___________/ \/___________/ \/__________/\/_/    \_\/    ║
║                                                                                            ║
║                    Retrieval Augmented Generation Query Interface                          ║
║                             by Jakob Bolliger © 2025 - AGPL-3.0                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝"""

# Subtitle text displayed below the banner
SUBTITLE = "Retrieval Augmented Generation Query Interface"

# Help text for the application
HELP_TEXT = """
KEYBINDINGS:
    Tab: Cycle between Prompt, Command, and Search modes
    F2: Show this help message
    Ctrl+J: Insert a newline in any input mode
    Enter: Execute command, submit prompt, or perform search
    Up/Down: Navigate through history (or navigate multiline input)
    Left/Right: Navigate text cursor or cycle through command suggestions
    Escape: Cancel command auto-completion
    Ctrl+C: Exit the application
    
SCROLLING:
    Shift+Up/Down: Scroll output faster (5 lines at a time)
    Page Up/Down: Scroll output very fast (20 lines at a time)
    Home/End: Jump to beginning/end of output
    
MODES:
    💬 Prompt Mode: Enter and edit your queries to the LLM
    🔧 Command Mode: Enter commands to manage chunks and sessions
    🔍 Search Mode: Directly search the vector database for content

COMMANDS:
    exit, quit, q: Exit the application
    help: View this help message
    
    CLEAR COMMANDS:
    clear: Clear all history and custom chunks
    ch, clear-history: Clear only conversation history
    cc, clear-chunks: Clear only custom chunks
    
    SAVE/LOAD:
    save: Save conversation history and custom chunks to a file
    
    CHUNK MANAGEMENT:
    add N, a N: Add chunk #N to your custom chunks list
    lc, list-chunks: Display your custom chunks list
    expand N [size], e N [size]: Expand chunk #N with optional context size
    set-chunks N, sc N: Set number of chunks to retrieve
    get-chunks, gc: Display current number of chunks to retrieve
    
    PROMPT COMMANDS:
    ap N, add-prompt N: Add custom chunk #N to your prompt
    p [query], prompt [query]: Send a prompt directly to the LLM
    
    SEARCH:
    s [query], search [query]: Search vector database for chunks
"""