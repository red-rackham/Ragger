import os
import argparse
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

# Default constants
DEFAULT_EMBEDDING_MODEL = "ibm-granite/granite-embedding-278m-multilingual"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_VECTORDB_DIR = Path("vector_dbs")

def load_documents(source_path):
    """Load markdown documents from a file or directory."""
    source_path = Path(source_path)
    documents = []
    
    if source_path.is_file():
        if source_path.suffix.lower() in ['.md', '.markdown']:
            loader = TextLoader(source_path)
            documents = loader.load()
    elif source_path.is_dir():
        loader = DirectoryLoader(
            source_path, 
            glob="**/*.md", 
            loader_cls=TextLoader,
            show_progress=True
        )
        documents = loader.load()
    
    print(f"Loaded {len(documents)} document(s) from {source_path}")
    return documents

def split_documents(documents, splitter_type="markdown", chunk_size=DEFAULT_CHUNK_SIZE, 
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split documents using the specified splitter."""
    if splitter_type.lower() == "markdown":
        # Markdown header splitter configuration
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
            ("####", "header_4"),
        ]
        
        # First split by headers
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = []
        
        for doc in documents:
            splits = markdown_splitter.split_text(doc.page_content)
            # Transfer metadata from parent doc to splits
            for split in splits:
                split.metadata.update(doc.metadata)
            md_header_splits.extend(splits)
            
        # If chunks are still large, apply recursive character splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(md_header_splits)
    else:  # Recursive splitter
        # Use recursive character splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
    
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks

def create_vectorstore(chunks, embedding_model_name, source_name, splitter_type, output_dir, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Create and save a vector store from document chunks."""
    # Setup embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
    print(f"Using embedding model: {embedding_model_name}")
    
    # Create vector store
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output path with naming convention
    model_name = embedding_model_name.split('/')[-1]
    db_name = f"{source_name}_{splitter_type}_{model_name}_cs{chunk_size}_co{chunk_overlap}"
    db_path = output_dir / db_name
    
    # Save vector store
    vectorstore.save_local(db_path)
    print(f"Vector store saved to: {db_path}")
    
    return db_path

def main():
    parser = argparse.ArgumentParser(description="Create vector store from markdown files")
    parser.add_argument("source", help="Path to markdown file or directory containing markdown files")
    parser.add_argument("--splitter", default="markdown", choices=["markdown", "recursive"], 
                      help="Document splitter type to use (default: markdown)")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL,
                      help=f"HuggingFace embedding model to use (default: {DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--output-dir", default=DEFAULT_VECTORDB_DIR,
                      help=f"Directory to save vector database (default: {DEFAULT_VECTORDB_DIR})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                      help=f"Size of text chunks (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                      help=f"Overlap between text chunks (default: {DEFAULT_CHUNK_OVERLAP})")
    
    args = parser.parse_args()
    
    # Determine source name (for db naming)
    source_path = Path(args.source)
    source_name = source_path.stem if source_path.is_file() else source_path.name
    
    # Process the documents
    documents = load_documents(args.source)
    if not documents:
        print(f"No markdown files found in {args.source}")
        return
    
    chunks = split_documents(
        documents, 
        args.splitter, 
        args.chunk_size,
        args.chunk_overlap
    )
    
    db_path = create_vectorstore(
        chunks, 
        args.embedding_model, 
        source_name, 
        args.splitter, 
        args.output_dir,
        args.chunk_size,
        args.chunk_overlap
    )
    
    print(f"\nVector database creation complete!")
    print(f"Source: {args.source}")
    print(f"Splitter: {args.splitter}")
    print(f"Embedding model: {args.embedding_model}")
    print(f"Vector database: {db_path}")

if __name__ == "__main__":
    main()