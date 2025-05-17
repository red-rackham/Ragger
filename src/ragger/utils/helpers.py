"""
Helper utilities for the Ragger application.

Contains helper functions used across the application.
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


def ensure_path_exists(path: str) -> Path:
    """Ensure the specified path exists.
    
    Args:
        path: Path to ensure exists
        
    Returns:
        Path object for the ensured path
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_filename(base_name: str, extension: str = ".txt") -> str:
    """Format a filename with sanitization.
    
    Args:
        base_name: Base name for the file
        extension: File extension
        
    Returns:
        Sanitized filename
    """
    # Remove invalid characters
    safe_name = re.sub(r'[\\/*?:"<>|]', "", base_name)
    # Replace spaces with underscores
    safe_name = re.sub(r'\s+', "_", safe_name)
    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = f".{extension}"
    return f"{safe_name}{extension}"