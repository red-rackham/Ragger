"""
Text formatting utilities for the terminal UI.

This module provides functions for formatting chunk displays and
other terminal display utilities that integrate with the UI.
"""
import shutil
from pathlib import Path

from ragger.ui.resources import Emojis


# Function moved to terminal package


def get_terminal_width() -> int:
    """Get the terminal width or a default value."""
    try:
        width = shutil.get_terminal_size().columns - 1
        # Ensure a minimum width to prevent formatting issues
        return max(width, 60)  # Minimum width of 60 characters
    except Exception:
        return 80  # fallback width


def get_chunk_header(doc, chunk_number: int) -> str:
    """Extract an appropriate header for a document chunk."""
    content = doc.page_content.strip()

    # Check for metadata headers (preferred source of headers)
    for key in doc.metadata:
        if key.startswith('header_'):
            return f"{Emojis.CHUNK} [#{chunk_number}] {doc.metadata[key]}"

    # Try to use first line as header if it's substantial
    lines = content.split('\n')
    if lines and (len(lines[0]) >= 20 or any(pattern in lines[0] for pattern in [":", ".", "?", "!", "-", "|"])):
        return f"{Emojis.CHUNK} [#{chunk_number}] {lines[0]}"

    # Default numbered chunk reference
    return f"{Emojis.CHUNK} [#{chunk_number}] Chunk {chunk_number}"


def get_simple_source_info(doc, chunk_number: int) -> str:
    """Get simplified source information for a document."""
    source = doc.metadata.get("source", "Unknown")
    chunk_index = f", chunk #{doc.metadata.get('chunk_index', chunk_number)}"

    # Display just the filename, not the full path
    if source != "Unknown":
        filename = Path(source).name  # Use Path for more reliable filename extraction
        return f"Source: {filename}{chunk_index}"
    else:
        return f"Source: {source}{chunk_index}"


def get_document_details(doc, chunk_number: int = None) -> str:
    """Extract useful details from a Document object."""
    # Basic document info
    details = [f"{Emojis.BULLET} Content Length: {len(doc.page_content)} characters"]

    # Display source information
    source_path = doc.metadata.get("source", "Unknown")
    original_chunk_index = doc.metadata.get('chunk_index', None)

    # Format source information
    if source_path != "Unknown":
        filename = Path(source_path).name
        details.append(f"{Emojis.BULLET} Source File: {filename}")
        details.append(f"{Emojis.BULLET} Full Path: {source_path}")
    else:
        details.append(f"{Emojis.BULLET} Source: {source_path}")

    # Add position information
    if chunk_number is not None:
        details.append(f"{Emojis.BULLET} Retrieval Position: #{chunk_number} of results")
    if original_chunk_index is not None:
        details.append(f"{Emojis.BULLET} Original Chunk Index: #{original_chunk_index} in source")

    # Document attributes - safely extract without multiple try/except blocks
    for attr_name, display_name in [
        ('id', 'Document ID'),
        ('lc_id', 'LC ID'),
        ('lc_attributes', 'LC Attributes'),
        ('type', 'Document Type')
    ]:
        if hasattr(doc, attr_name) and getattr(doc, attr_name):
            try:
                details.append(f"{Emojis.BULLET} {display_name}: {getattr(doc, attr_name)}")
            except Exception:
                pass

    # Special callable attributes
    if hasattr(doc, 'get_lc_namespace') and callable(getattr(doc, 'get_lc_namespace')):
        try:
            namespace = doc.get_lc_namespace()
            details.append(f"{Emojis.BULLET} LC Namespace: {namespace}")
        except Exception:
            pass

    if hasattr(doc, 'is_lc_serializable'):
        try:
            if doc.is_lc_serializable:
                details.append(f"{Emojis.BULLET} LC Serializable: Yes")
        except Exception:
            pass

    # Display metadata excluding source which we've already shown
    metadata_entries = [(k, v) for k, v in doc.metadata.items() if k != "source"]
    if metadata_entries:
        details.append(f"{Emojis.BULLET} Document Metadata:")
        for key, value in metadata_entries:
            details.append(f"  - {key}: {value}")

    # Show similarity score if available
    score = doc.metadata.get("score", None)
    if score is not None:
        details.append(f"{Emojis.BULLET} Relevance Score: {score:.4f}")

    # Display document hierarchy information
    headers = [key for key in doc.metadata.keys() if key.startswith("header_")]
    if headers:
        details.append(f"{Emojis.BULLET} Document Hierarchy:")
        for header in sorted(headers):
            level = header.split("_")[1]
            details.append(f"  - Level {level}: {doc.metadata[header]}")

    # Pydantic model data if available
    if hasattr(doc, 'model_dump') and callable(getattr(doc, 'model_dump')):
        try:
            model_info = doc.model_dump()
            extra_keys = [k for k in model_info.keys() if k not in ('metadata', 'page_content')]
            if extra_keys:
                details.append(f"{Emojis.BULLET} Model Data:")
                for key in extra_keys:
                    details.append(f"  - {key}: {model_info[key]}")
        except Exception:
            pass

    return "\n".join(details)


def format_chunk_display(doc, chunk_number: int, full_chunks: bool, terminal_width: int) -> str:
    """Format document chunk for display with header and details."""
    chunk_text = ""
    content = doc.page_content.strip()

    # Get and add header
    header = get_chunk_header(doc, chunk_number)
    chunk_text += f"{header}\n\n"

    # Add content, handling the case where first line was used as header
    if header.endswith(content.split('\n')[0]):
        lines = content.split('\n')
        if len(lines) > 1:
            remaining_content = '\n'.join(lines[1:]).strip()
            chunk_text += f"{remaining_content}\n\n"
    else:
        chunk_text += f"{content}\n\n"

    # Add chunk information
    if full_chunks:
        chunk_text += f"{Emojis.DETAILS} CHUNK #{chunk_number} DETAILS:\n"
        chunk_text += get_document_details(doc, chunk_number) + "\n"
    else:
        # Simple source info for non-full mode
        chunk_text += get_simple_source_info(doc, chunk_number) + "\n"

    return chunk_text


def expand_chunk_context(chunk, source_path, chunk_index, terminal_width, context_size=500):
    """Expand a chunk to see more context from its source document."""
    try:
        # First check if the source file exists and is readable
        source_file = Path(source_path)
        if not source_file.exists():
            return f"Error: Source file not found: {source_path}"

        # Determine the file type
        file_extension = source_file.suffix.lower()

        # Strategy depends on file type
        if file_extension in ('.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.yaml', '.yml'):
            # For text-based files, read the entire content
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find the chunk's content in the file
            chunk_content = chunk.page_content.strip()
            if chunk_content in content:
                # Get context window (context_size chars before and after)
                start_pos = max(0, content.find(chunk_content) - context_size)
                end_pos = min(len(content), content.find(chunk_content) + len(chunk_content) + context_size)

                # Extract expanded context
                expanded_context = content[start_pos:end_pos]

                # Format the result with header and dashed border
                result = f"Expanded context from {source_file.name} around chunk #{chunk_index}:\n"
                result += "┄" * 80 + "\n\n"

                # Show the chunk position within the expanded context
                if start_pos > 0:
                    result += "... (content omitted) ...\n\n"

                # Mark the original chunk content within the expanded view
                chunk_start = expanded_context.find(chunk_content)
                if chunk_start >= 0:
                    before_chunk = expanded_context[:chunk_start]
                    after_chunk = expanded_context[chunk_start + len(chunk_content):]

                    result += before_chunk
                    result += f"\n{'┄' * 20} ORIGINAL CHUNK START {'┄' * 20}\n"
                    result += chunk_content
                    result += f"\n{'┄' * 20} ORIGINAL CHUNK END {'┄' * 20}\n"
                    result += after_chunk
                else:
                    # Fallback if chunk location can't be identified
                    result += expanded_context

                if end_pos < len(content):
                    result += "\n\n... (content omitted) ..."

                return result
            else:
                # If exact match not found, return entire file with warning and dashed border
                warning = f"Note: Could not locate exact chunk in source. " \
                          f"Displaying full content of {source_file.name}:\n"
                warning += "┄" * 80 + "\n\n"
                return warning + content
        else:
            # For non-text files, return information about the file with dashed border
            warning = f"Source file type ({file_extension}) is not supported for expansion. " \
                      f"Please check the original file at {source_path}.\n"
            warning += "┄" * 80 + "\n"
            return warning

    except Exception as e:
        error_msg = f"Error expanding chunk context: {str(e)}\n"
        error_msg += "┄" * 80 + "\n"
        return error_msg