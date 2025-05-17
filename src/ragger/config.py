"""
Configuration settings for the ragger application.
All constants and configuration variables are defined here.
"""
from pathlib import Path

# LLM and embedding settings
DEFAULT_EMBEDDING_MODEL = "ibm-granite/granite-embedding-278m-multilingual"
DEFAULT_VECTORDB_DIR = Path(".")
DEFAULT_LLM_MODEL = "mistral"
DEFAULT_NUM_CHUNKS = 5
STOP_SEQUENCE = ["<end_of_turn>"]
LLM_TIMEOUT = 60

# Ollama settings
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# System prompt components
# These can be combined as needed based on whether history is available
SYSTEM_PROMPT_INTRO = """You are my personal assistant that helps me with
researching and answering prompts based on the provided context.

Respond only based on the context provided and avoid using prior
knowledge.

It is usually expected that in your response you reference the
chunks like this (reference)
"""

SYSTEM_PROMPT_HISTORY = """
Previous conversation:
{history}
"""

SYSTEM_PROMPT_CONTEXT = """
Context:
{context}
"""

SYSTEM_PROMPT_QUERY = """
Prompt:
{input}

Response:
"""

# Default full system prompt (without history)
DEFAULT_SYSTEM_PROMPT = (
    SYSTEM_PROMPT_INTRO + 
    SYSTEM_PROMPT_CONTEXT + 
    SYSTEM_PROMPT_QUERY
)

