"""
Conversation handling utilities for the Ragger application.

Contains functions for saving and loading conversation history.
"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from ragger.utils.helpers import ensure_path_exists, format_filename


def save_conversation_history(conversation_history: List[Tuple[str, str]],
                             retrieved_chunks_history: List[Any] = None,
                             model_name: str = "unknown",
                             filepath: Optional[str] = None) -> str:
    """Save conversation history to a file.
    
    Args:
        conversation_history: List of (prompt, response) tuples
        retrieved_chunks_history: Optional list of retrieved chunks for each turn
        model_name: Name of the model used
        filepath: Optional file path to save to
        
    Returns:
        Path to the saved file
    """
    # If no filepath specified, create one based on timestamp and model
    if not filepath:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = format_filename(f"conversation_{model_name}_{timestamp}", ".json")
        filepath = str(Path("conversations") / filename)
    
    # Ensure the directory exists
    ensure_path_exists(str(Path(filepath).parent))
    
    # Prepare conversation data
    data = {
        "model": model_name,
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "turns": []
    }
    
    # Add each conversation turn
    for i, (prompt, response) in enumerate(conversation_history):
        turn_data = {
            "turn": i + 1,
            "prompt": prompt,
            "response": response
        }
        
        # Add chunk information if available (excluding non-serializable attributes)
        if retrieved_chunks_history and i < len(retrieved_chunks_history):
            chunks_data = []
            for chunk in retrieved_chunks_history[i]:
                chunk_data = {
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                }
                chunks_data.append(chunk_data)
            turn_data["chunks"] = chunks_data
            
        data["turns"].append(turn_data)
    
    # Write to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return filepath