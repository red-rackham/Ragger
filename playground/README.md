# 🧪 Ragger Playground

Utility scripts for document processing and vector database creation.

## 📄 Document Processing

- **pdf_to_accessible_markdown.py**: Creates accessible markdown by replacing images with text descriptions.
  ```bash
  python pdf_to_accessible_markdown.py input_dir [output_dir]
  ```

- **pdf_to_annotated_documents.py**: Processes PDFs with image descriptions as annotations.
  ```bash
  python pdf_to_annotated_documents.py input_dir [output_dir]
  ```

- **pdf_batch_processor.py**: Batch processes PDFs with image annotations.
  ```bash
  python pdf_batch_processor.py input_dir [output_dir]
  ```

## 🔄 Format Conversion

- **json_to_md_with_descriptions.py**: Converts docling JSON to markdown with image descriptions.
  ```bash
  python json_to_md_with_descriptions.py input_path [output_dir]
  ```

## 🔍 Vector Database Creation

- **markdown_to_vector.py**: Creates vector databases from markdown files.
  ```bash
  python markdown_to_vector.py source_path [--options]
  ```
  Key options:
  - `--splitter`: "markdown" or "recursive" (default: markdown)
  - `--embedding-model`: HuggingFace model name
  - `--chunk-size`: Text chunk size
  - `--output-dir`: Save location

## 🤖 Testing

- **ollama_test.py**: Simple test for Ollama LLM integration.
  ```bash
  python ollama_test.py
  ```

## Example: Create Vector Database

```bash
# Create a vector database with markdown-aware splitting
python markdown_to_vector.py docs/ --chunk-size 800 --chunk-overlap 100
```

## Example: Process PDFs

```bash
# Process PDFs and create accessible markdown
python pdf_to_accessible_markdown.py pdf_docs/ output/
```