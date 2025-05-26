"""
Configuration settings for the ragger application.
All constants and configuration variables are defined here.
"""
from pathlib import Path

# Context and Content Settings
DEFAULT_CONTEXT_SIZE = 500
CHUNK_PREVIEW_LENGTH_SHORT = 50
CHUNK_PREVIEW_LENGTH_MEDIUM = 100
CHUNK_SEPARATOR_LINE_LENGTH = 40

# Terminal and UI Settings  
TERMINAL_MIN_WIDTH = 60
TERMINAL_FALLBACK_WIDTH = 80
UI_SCROLL_STEP_SMALL = 5
UI_SCROLL_STEP_LARGE = 40
UI_MAX_NAVIGATION_ITERATIONS = 10000
UI_MODE_INDICATOR_WIDTH = 3
UI_INPUT_WINDOW_HEIGHT = 3
UI_COMPLETION_MENU_MAX_HEIGHT = 5
UI_MAX_COMPLETIONS = 10
UI_MIN_REDRAW_INTERVAL = 0.05

# Threading and Performance
LLM_RESULT_CHECK_INTERVAL = 0.1
THREAD_SHUTDOWN_TIMEOUT = 5.0

# Error Message Formatting
ERROR_MESSAGE_PREFIX = "\n🚫 Error: "
WARNING_MESSAGE_PREFIX = "\n⚠️  Warning: "
INFO_MESSAGE_PREFIX = "\n💡 "

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

