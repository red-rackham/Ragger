# Ragger

A practical terminal-based interface for Retrieval-Augmented Generation (RAG) queries.

## About

Ragger is a simple terminal-based interface for RAG (Retrieval-Augmented Generation) queries. It allows you to search through your document collection and receive responses from large language models that are grounded in your data. The tool provides a straightforward way to interact with your knowledge base through a text-based interface.

## Features

- **RAG-powered search**: Query your documents and get LLM responses based on retrieved content
- **Chunk management**: Save and reuse relevant chunks of information
- **Conversation history**: Keep track of your query sessions
- **Terminal UI**: Simple command and prompt modes

## Prerequisites

- Python 3.12 or higher
- [Ollama](https://ollama.ai) for local LLM inference
- A vector database containing your documents

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

1. **Add a chunk to your custom list**:
   ```
   🔧 add 2
   ```
   This adds chunk #2 from the most recent search results to your personal archive.

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
| `-e, --embedding-model` | HuggingFace embedding model to use |
| `-l, --llm-model` | Ollama LLM model to use |
| `-sc, --set-chunks` | Number of chunks to retrieve (default: 5) |
| `-d, --hide-chunks` | Hide the retrieved text chunks |
| `-f, --full-chunks` | Show full chunk content and all metadata |
| `-s, --save` | Save conversation history to file when exiting |
| `--debug` | Enable debug mode for troubleshooting UI issues |

## Commands Reference

### RAG Commands
- `a N, add N`: Add chunk #N to your custom chunks list
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