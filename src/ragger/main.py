"""
Copyright (c) 2025 Jakob Bolliger

This file is part of Ragger.

This program is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)
The full text of the license can be found in the LICENSE file in the
root directory of this source tree.

Ragger - Main entry point for the RAG-based interactive query application.
"""
import argparse
import sys

from ragger.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL, DEFAULT_NUM_CHUNKS, DEFAULT_VECTORDB_DIR
from ragger.ui.resources import Emojis
from ragger.core.rag import RagService
from ragger.core.commands import CommandManager
from ragger.interfaces import CliInterface


def main():
    """Main entry point for the application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Interactive query session with a vector database")
    parser.add_argument("db_path", nargs='?', default=str(DEFAULT_VECTORDB_DIR),
                        help=f"Path to the vector database (default: {DEFAULT_VECTORDB_DIR})")
    parser.add_argument("-e", "--embedding-model", default=None,
                        help=f"HuggingFace embedding model (default: auto-detect from ragger.info, fallback: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("-l", "--llm-model", default=DEFAULT_LLM_MODEL,
                        help=f"Ollama LLM model to use (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("-sc", "--set-chunks", type=int, default=DEFAULT_NUM_CHUNKS,
                        help=f"Number of chunks to retrieve (default: {DEFAULT_NUM_CHUNKS})")
    parser.add_argument("-d", "--hide-chunks", action="store_true",
                        help="Hide the retrieved text chunks")
    parser.add_argument("-f", "--full-chunks", action="store_true",
                        help="Show full chunk content and all metadata")
    parser.add_argument("-s", "--save", action="store_true",
                        help="Save conversation history to file when exiting")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for troubleshooting scrolling and other UI issues")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging for debugging")

    args = parser.parse_args()

    print(f"\nInitializing Ragger - Copyright (c) 2025 Jakob Bolliger - AGPL License")
    print(f"{Emojis.WAITING} Loading vector database from {args.db_path}, please wait...")

    try:
        # Initialize core services
        rag_service = RagService(
            args.db_path,
            args.embedding_model,
            args.llm_model,
            args.set_chunks
        )

        command_manager = CommandManager()

        # Create and run CLI interface
        cli = CliInterface(rag_service, command_manager, debug_mode=args.debug, verbose=args.verbose)

        # Run the interactive session
        cli.interactive_session(
            hide_chunks=args.hide_chunks,
            full_chunks=args.full_chunks,
            auto_save=args.save
        )
    except KeyboardInterrupt:
        # Try to save conversation history before exiting if auto-save is enabled
        if hasattr(args, 'save') and args.save and 'rag_service' in locals():
            rag_service = locals()['rag_service']
            if hasattr(rag_service, 'conversation_history') and rag_service.conversation_history:
                try:
                    result = rag_service.save_history()
                    if result['success']:
                        print(f"\n{Emojis.CHECK} Conversation history saved to {result['filepath']}")
                    else:
                        print(f"\n{Emojis.ERROR} Error saving conversation history: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"\n{Emojis.ERROR} Error saving conversation history: {str(e)}")

        print("\n\nSession terminated by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()