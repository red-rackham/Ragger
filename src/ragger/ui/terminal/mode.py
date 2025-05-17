"""
Input mode management for the terminal UI.

This module provides the InputMode enum to track and manage different
input modes in the terminal interface.
"""
import enum


class InputMode(enum.Enum):
    """Enumeration of input modes for the CLI interface."""
    PROMPT = "prompt"
    COMMAND = "command"
    SEARCH = "search"