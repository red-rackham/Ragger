# Ragger

A practical terminal-based interface for Retrieval-Augmented Generation (RAG) queries.

## About

Ragger is a simple terminal-based interface for RAG (Retrieval-Augmented Generation) queries. It allows you to search through your document collection and receive responses from large language models that are grounded in your data. The tool provides a straightforward way to interact with your knowledge base through a text-based interface.

## Features

- **RAG-powered search**: Query your documents and get LLM responses based on retrieved content
- **Auto-detection**: Automatically detects embedding models from existing vector databases
- **Search-only mode**: Use as a semantic search tool without requiring an LLM
- **Chunk management**: Save and reuse relevant chunks of information
- **Conversation history**: Keep track of your query sessions
- **Terminal UI**: Simple command and prompt modes with helpful guidance

## Prerequisites

- Python 3.12 or higher
- [Ollama](https://ollama.ai) for local LLM inference (optional for search-only mode)
- A vector database containing your documents
- GPU support recommended for large vector databases

## Installation

```bash
# Clone the repository
... (todo)
cd ragger

# Install in development mode
pip install -e .
```

## Quick Start

1. **Start the application**:
   ```bash
   ragger /path/to/your/vectordb
   ```

2. **Enter a query in prompt mode**:
   ```
   💬 How does retrieval augmented generation work?
   ```

3. **Use commands in command mode (press Tab to switch)**:
   ```
   🔧 help
   ```

## Working with Chunks

Chunks are the building blocks of RAG. Here's how to work with them in Ragger:

### Chunk Behavior Modes

Ragger offers intelligent chunk selection with three modes:

#### **Default Smart Behavior** (Recommended)
- **When you have custom chunks**: Uses only your saved custom chunks (no search)
- **When you have no custom chunks**: Performs normal vector database search
- This provides the best user experience by automatically adapting to your workflow

#### **Force Search Mode** (`--ignore-custom` or `-i`)
```bash
ragger /path/to/vectordb --ignore-custom
```
- Always searches the vector database, even if you have custom chunks
- Useful when you want fresh results regardless of your saved chunks

#### **Combined Mode** (`--combine-chunks`)
```bash
ragger /path/to/vectordb --combine-chunks
```
- Searches the vector database AND includes your custom chunks
- Provides maximum context by combining both sources
- Custom chunks appear first, followed by unique search results

**Example Usage:**
```bash
# Smart default behavior (recommended)
ragger /path/to/vectordb

# Always search, ignore any custom chunks
ragger /path/to/vectordb -i

# Search and include custom chunks together
ragger /path/to/vectordb --combine-chunks
```

### Viewing Retrieved Chunks

When you enter a query, Ragger automatically retrieves relevant chunks from your vector database:

```
💬 Tell me about machine learning algorithms
```

The system will display:
```
🔍 Searching knowledge base...
📚 Retrieved chunks:
```

Followed by numbered chunks from your documents that match your query.

### Managing Custom Chunks

You can save chunks for later use:

1. **Add chunk(s) to your custom list**:
   ```
   🔧 add 2
   🔧 add 2 3 5
   ```
   This adds chunk #2 from the most recent search results, or multiple chunks #2, #3, and #5 to your personal archive.

2. **List your saved chunks**:
   ```
   🔧 lc
   ```
   This displays all chunks you've saved.

3. **Add a custom chunk to your prompt**:
   ```
   🔧 ap 1
   ```
   This includes custom chunk #1 with your next query to provide additional context.

4. **Expand a chunk to see more context**:
   ```
   🔧 expand 3 5
   ```
   This shows chunk #3 with 5 additional lines of context before and after.

5. **Clear your custom chunks**:
   ```
   🔧 cc
   ```
   This removes all saved chunks.

### Search Commands

Searching is a core functionality of Ragger:

1. **Direct search without using history**:
   ```
   🔧 search neural networks
   ```
   This searches for "neural networks" without sending the query to the LLM.

2. **Set the number of chunks to retrieve**:
   ```
   🔧 set-chunks 8
   ```
   This changes how many chunks are retrieved per query.

3. **Get current chunk setting**:
   ```
   🔧 get-chunks
   ```
   This shows the current number of chunks being retrieved per query.

## Command Line Options

| Option | Description |
|--------|-------------|
| `-e, --embedding-model` | HuggingFace embedding model to use (auto-detected if available) |
| `-l, --llm-model` | Ollama LLM model to use (optional for search-only mode) |
| `-sc, --set-chunks` | Number of chunks to retrieve (default: 5) |
| `-i, --ignore-custom` | Always search vector database, ignore custom chunks |
| `--combine-chunks` | Search vector database AND include custom chunks |
| `-d, --hide-chunks` | Hide the retrieved text chunks |
| `-f, --full-chunks` | Show full chunk content and all metadata |
| `-s, --save` | Save conversation history to file when exiting |
| `--debug` | Enable debug mode for troubleshooting UI issues |

## Commands Reference

### RAG Commands
- `a N [N2 N3...], add N [N2 N3...]`: Add chunk(s) to your custom chunks list
- `e N [size], expand N [size]`: Expand chunk #N with optional context size
- `ap N, add-prompt N`: Add custom chunk #N to your prompt
- `s [query], search [query]`: Search vector database for chunks
- `sc N, set-chunks N`: Set number of chunks to retrieve
- `gc, get-chunks`: Display current number of chunks to retrieve
- `lc, list-chunks`: Display your custom chunks list

### History Management
- `history`: Show conversation history
- `ch, clear-history`: Clear only conversation history
- `cc, clear-chunks`: Clear only custom chunks
- `clear`: Clear all history and custom chunks
- `save`: Save conversation history to a file

### Navigation
- `exit, quit, q`: Exit the application
- `help, h, ?`: View the help message

### Keyboard Shortcuts
- `Tab`: Cycle between Prompt, Command, and Search modes
- `F2`: Show help message
- `Ctrl+J`: Insert a newline in any input mode
- `Up/Down`: Navigate through history or multiline input
- `Left/Right`: Navigate cursor or cycle through command suggestions
- `Escape`: Cancel command auto-completion
- `Ctrl+C`: Exit the application

## Interface Modes

- **💬 Prompt Mode**: Enter and edit your queries to the LLM
- **🔧 Command Mode**: Enter commands to manage chunks and sessions
- **🔍 Search Mode**: Directly search the vector database for content

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
The full text of the license can be found in the LICENSE file in the root directory of this source tree.

## Copyright

Copyright (c) 2025 Jakob Bolliger