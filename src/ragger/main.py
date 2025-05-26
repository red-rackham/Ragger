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

from ragger.config import (DEFAULT_EMBEDDING_MODEL, DEFAULT_LLM_MODEL,
                           DEFAULT_NUM_CHUNKS, DEFAULT_VECTORDB_DIR)
from ragger.core.commands import CommandManager
from ragger.core.exceptions import (EmbeddingModelError, VectorStoreError,
                                    VectorStoreLoadError,
                                    VectorStoreNotFoundError)
from ragger.core.rag import RagService
from ragger.interfaces import CliInterface
from ragger.ui.resources import Emojis, format_error_message


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
    parser.add_argument("-i", "--ignore-custom", action="store_true",
                        help="Always search vector database, ignore custom chunks")
    parser.add_argument("--combine-chunks", action="store_true",
                        help="Search vector database AND include custom chunks")

    args = parser.parse_args()

    # Validate conflicting chunk behavior flags
    if args.ignore_custom and args.combine_chunks:
        print(format_error_message("Cannot use --ignore-custom and --combine-chunks together"))
        sys.exit(1)

    print(f"\nInitializing Ragger - Copyright (c) 2025 Jakob Bolliger - AGPL License")
    print(f"{Emojis.WAITING} Loading vector database from {args.db_path}, please wait...")

    try:
        # Initialize core services
        rag_service = RagService(
            args.db_path,
            args.embedding_model,
            args.llm_model,
            args.set_chunks,
            ignore_custom=args.ignore_custom,
            combine_chunks=args.combine_chunks
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
    except VectorStoreNotFoundError as e:
        print(f"\n{Emojis.ERROR} Vector Database Not Found")
        print(f"Could not find vector database at: {e.path}")
        print(f"\nPlease ensure the path exists and contains a valid FAISS vector database.")
        print(f"Expected files: index.faiss, index.pkl")
        if args.verbose:
            print(f"\nDetailed error: {str(e)}")
        sys.exit(1)
    except EmbeddingModelError as e:
        print(f"\n{Emojis.ERROR} Embedding Model Error")
        print(f"Failed to load embedding model: {e.model_name}")
        print(f"\nThis could be due to:")
        print(f"  • Network connectivity issues")
        print(f"  • Invalid model name") 
        print(f"  • Missing HuggingFace dependencies")
        if args.verbose:
            print(f"\nDetailed error: {str(e)}")
        sys.exit(1)
    except VectorStoreLoadError as e:
        print(f"\n{Emojis.ERROR} Vector Database Load Error")
        print(f"Failed to load vector database from: {e.path}")
        print(f"Embedding model: {e.embedding_model}")
        print(f"\nThis could be due to:")
        print(f"  • Incompatible embedding model (database was created with different model)")
        print(f"  • Corrupted database files")
        print(f"  • Version compatibility issues")
        if args.verbose:
            print(f"\nDetailed error: {str(e)}")
        sys.exit(1)
    except VectorStoreError as e:
        print(f"\n{Emojis.ERROR} Vector Database Error")
        print(f"An error occurred with the vector database: {str(e)}")
        if args.verbose and hasattr(e, '__cause__') and e.__cause__:
            print(f"\nDetailed error: {str(e.__cause__)}")
        sys.exit(1)
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