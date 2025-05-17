"""
Text manipulation utilities for terminal display.

This module provides core text manipulation functions and formatting utilities
used by the terminal UI for displaying text, wrapping content, and drawing boxes.
"""
from ragger.ui.resources import BORDER_STYLES


def pad_right(text: str, target_width: int) -> str:
    """Pad text with spaces to reach target width, compensating for emoji width.

    Terminal displays often render emoji as double-width characters but Python
    counts them as single characters. This function compensates for that discrepancy.

    Args:
        text: The text to pad
        target_width: The desired width of the padded text

    Returns:
        Right-padded text with proper emoji width compensation
    """
    # Count emojis in the text - emojis typically occupy double the visual width
    # Most common emoji ranges: Basic (U+1F000-U+1FFFF), Supplemental Symbols, etc.
    emoji_count = sum(1 for char in text if (
            0x1F000 <= ord(char) <= 0x1FFFF or  # Basic Emoji
            0x2600 <= ord(char) <= 0x27BF or  # Misc symbols and pictographs
            0x2300 <= ord(char) <= 0x23FF or  # Misc Technical
            0x2B50 <= ord(char) <= 0x2B55  # Additional emoji
    ))

    # Calculate current width with emoji compensation (1 extra space per emoji)
    current_width = len(text) + emoji_count

    # Add padding if needed
    return text + ' ' * (target_width - current_width) if current_width < target_width else text


def wrap_line(line: str, available_width: int, vertical: str) -> list:
    """Wrap a single line of text to fit within available width.

    Args:
        line: Text line to wrap
        available_width: Available width for content
        vertical: Vertical border character

    Returns:
        List of formatted wrapped lines
    """
    wrapped_lines = []
    remaining = line

    while remaining:
        # If remaining text fits, just return it
        if len(remaining) <= available_width:
            padded_content = pad_right(remaining, available_width)
            wrapped_lines.append(f"{vertical} {padded_content} {vertical}")
            break

        # Find the last space within the available width
        split_at = available_width
        while 0 < split_at < len(remaining) and remaining[split_at - 1] != ' ':
            split_at -= 1

        # If no space found, force split at available_width
        if split_at == 0 or split_at >= len(remaining):
            split_at = min(available_width, len(remaining))

        # Format the chunk with right border
        chunk = remaining[:split_at].rstrip()
        padded_content = pad_right(chunk, available_width)
        wrapped_lines.append(f"{vertical} {padded_content} {vertical}")

        # Update remaining text
        remaining = remaining[split_at:].lstrip() if split_at < len(remaining) else ''

    return wrapped_lines


def get_wrapped_text_in_box(text: str, terminal_width: int, title: str = None, border_style: str = "solid") -> str:
    """Create text wrapped in a box with proper word wrapping and borders.
    
    Args:
        text: The text to display in the box
        terminal_width: The width of the terminal in characters
        title: Optional title to display above the box
        border_style: Box style - "solid" for normal borders, "dashed" for dashed borders
        
    Returns:
        String with the formatted text box
    """
    # Get border style characters
    style = BORDER_STYLES.get(border_style, BORDER_STYLES["solid"])
    
    # Prepare result lines
    result_lines = []
    
    # Add title if provided
    if title:
        result_lines.append(f"{title}")
        
    # Add top of box
    result_lines.append(f"{style['top_left']}{style['horizontal'] * (terminal_width - 2)}{style['top_right']}")
    
    # Calculate available width for content
    available_width = terminal_width - 4  # -4 for borders and spacing
    
    # Process each line
    for line in text.split('\n'):
        # If line is longer than available width, wrap it
        if len(line) > available_width:
            wrapped_lines = wrap_line(line, available_width, style['vertical'])
            for wrapped_line in wrapped_lines:
                result_lines.append(wrapped_line)
        else:
            # Add line with both borders ensuring consistent spacing
            padded_content = pad_right(line, available_width)
            result_lines.append(f"{style['vertical']} {padded_content} {style['vertical']}")
    
    # Add bottom of box
    result_lines.append(f"{style['bottom_left']}{style['horizontal'] * (terminal_width - 2)}{style['bottom_right']}")
    
    # Return the result as a single string
    return "\n".join(result_lines)


def print_wrapped_text_in_box(text: str, terminal_width: int, title: str = None, border_style: str = "solid") -> None:
    """Print text wrapped in a box with proper word wrapping and borders.

    Handles long lines by wrapping at word boundaries and preserves box formatting with proper borders.
    Automatically adjusts for emoji width discrepancies in terminal display.

    Args:
        text: The text to display in the box
        terminal_width: The width of the terminal in characters
        title: Optional title to display above the box
        border_style: Box style - "solid" for normal borders, "dashed" for dashed borders
    """
    # Get the formatted box text
    formatted_text = get_wrapped_text_in_box(text, terminal_width, title, border_style)
    
    # Print the formatted text
    print(formatted_text)