# Ragger

A RAG-based interactive query application.

## About

Ragger is a terminal-based interface for RAG (Retrieval-Augmented Generation) queries. It allows you to search through your document collection and get responses from an LLM that are grounded in your data.

## Installation

Install in development mode:

```bash
pip install -e .
```

## Usage

Run the application:

```bash
ragger
```

Or as a module:

```bash
python -m ragger
```

## Command Line Options

- `-e, --embedding-model`: HuggingFace embedding model to use
- `-l, --llm-model`: Ollama LLM model to use
- `-n, --num-chunks`: Number of chunks to retrieve
- `-d, --hide-chunks`: Hide the retrieved text chunks
- `-f, --full-chunks`: Show full chunk content and all metadata
- `-s, --save`: Save conversation history to file when exiting

## License

This project is licensed under the terms of the license found in the LICENSE file in the root directory of this source tree.

## Copyright

Copyright (c) [year] [copyright holder]. All rights reserved.

## Contributing

Contributions are welcome. All contributed code must be licensed under the same license as the project.