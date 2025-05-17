#!/usr/bin/env python3
"""
Test script for prompt-toolkit multicolumn completion
"""
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# Sample commands to complete
commands = [
    "exit", "quit", "help", "history", "clear", 
    "clear-history", "clear-chunks", "add", "expand",
    "list-chunks", "search", "set-chunks", "get-chunks"
]

# Create completer
completer = WordCompleter(commands, ignore_case=True)

# Import complete style enum
from prompt_toolkit.shortcuts import CompleteStyle

# Prompt with multi-column display
result = prompt('Command: ', 
                completer=completer, 
                complete_style=CompleteStyle.MULTI_COLUMN)
print(f"You entered: {result}")