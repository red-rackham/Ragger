import sys
import logging
import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Default constants
DEFAULT_EMBEDDING_MODEL = "ibm-granite/granite-embedding-278m-multilingual"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_VECTORDB_DIR = Path("vector_dbs")

def setup_logging(verbose=False):
    """Setup comprehensive logging with configurable levels."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('markdown_to_vector.log', mode='a')
        ]
    )
    
    # Configure third-party library logging
    if verbose:
        logging.getLogger('langchain').setLevel(logging.DEBUG)
        logging.getLogger('langchain_community').setLevel(logging.DEBUG)
        logging.getLogger('transformers').setLevel(logging.INFO)
        logging.getLogger('sentence_transformers').setLevel(logging.INFO)
        logging.getLogger('faiss').setLevel(logging.DEBUG)
    else:
        logging.getLogger('langchain').setLevel(logging.WARNING)
        logging.getLogger('langchain_community').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
        logging.getLogger('faiss').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def load_documents(source_path):
    """Load markdown documents from a file or directory."""
    logger = logging.getLogger(__name__)
    source_path = Path(source_path)
    documents = []
    
    logger.info(f"Starting document loading from: {source_path}")
    
    try:
        if source_path.is_file():
            if source_path.suffix.lower() in ['.md', '.markdown']:
                logger.info(f"Loading single markdown file: {source_path}")
                loader = TextLoader(source_path)
                start_time = time.time()
                documents = loader.load()
                load_time = time.time() - start_time
                logger.info(f"File loaded in {load_time:.2f} seconds")
            else:
                logger.warning(f"File {source_path} is not a markdown file")
        elif source_path.is_dir():
            logger.info(f"Loading markdown files from directory: {source_path}")
            loader = DirectoryLoader(
                source_path, 
                glob="**/*.md", 
                loader_cls=TextLoader,
                show_progress=True
            )
            start_time = time.time()
            documents = loader.load()
            load_time = time.time() - start_time
            logger.info(f"Directory loaded in {load_time:.2f} seconds")
        else:
            logger.error(f"Source path {source_path} does not exist")
            return []
    except Exception as e:
        logger.error(f"Error loading documents from {source_path}: {str(e)}")
        logger.exception("Full traceback:")
        return []
    
    logger.info(f"Successfully loaded {len(documents)} document(s) from {source_path}")
    for i, doc in enumerate(documents[:5]):  # Log first 5 docs
        logger.debug(f"Document {i}: {doc.metadata.get('source', 'unknown')} - {len(doc.page_content)} chars")
    
    if len(documents) > 5:
        logger.debug(f"... and {len(documents) - 5} more documents")
    
    return documents

def split_documents(documents, splitter_type="markdown", chunk_size=DEFAULT_CHUNK_SIZE, 
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Split documents using the specified splitter."""
    logger = logging.getLogger(__name__)
    logger.info(f'Starting document splitting with {splitter_type} splitter')
    logger.info(f'Parameters: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}')
    
    start_time = time.time()
    
    try:
        if splitter_type.lower() == "markdown":
            logger.info('Using Markdown header splitter followed by recursive character splitter')
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
            
            for i, doc in enumerate(documents):
                doc_source = doc.metadata.get('source', 'unknown')
                logger.info(f"Processing document {i+1}/{len(documents)}: {doc_source}")
                logger.debug(f"Document length: {len(doc.page_content)} characters")
                
                try:
                    doc_start_time = time.time()
                    splits = markdown_splitter.split_text(doc.page_content)
                    doc_time = time.time() - doc_start_time
                    
                    # Transfer metadata from parent doc to splits
                    for split in splits:
                        split.metadata.update(doc.metadata)
                    md_header_splits.extend(splits)
                    
                    logger.info(f"Document {doc_source} split into {len(splits)} header sections in {doc_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error splitting document {doc_source}: {str(e)}")
                    logger.exception("Full traceback:")
                    continue
                
            logger.info(f"Header splitting complete: {len(md_header_splits)} header sections created")
            
            # Apply recursive character splitter to ensure consistent chunk sizes
            # (markdown splitter only splits on headers, sections can still be very large)
            logger.info("Applying recursive character splitter to ensure consistent chunk sizes")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(md_header_splits)
        else:
            logger.info('Using recursive character splitter only')
            # Use only recursive character splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = text_splitter.split_documents(documents)
    
    except Exception as e:
        logger.error(f"Error during document splitting: {str(e)}")
        logger.exception("Full traceback:")
        return []
    
    total_time = time.time() - start_time
    logger.info(f"Document splitting completed in {total_time:.2f} seconds")
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Log chunk size statistics
    if chunks:
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        logger.info(f"Chunk size stats - Avg: {avg_size:.0f}, Min: {min_size}, Max: {max_size}")
    
    return chunks

def create_vectorstore(chunks, embedding_model_name, source_name, splitter_type, output_dir, device="auto",
                       chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP, source_document_count=None):
    """Create and save a vector store from document chunks."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting vectorstore creation with {len(chunks)} chunks")
    
    start_time = time.time()
    
    try:
        # Setup embedding model
        logger.info(f"Initializing embedding model: {embedding_model_name}")
        model_start_time = time.time()
        
        # Configure device for SentenceTransformer
        if device == "auto":
            # SentenceTransformer will automatically use available GPUs
            model_kwargs = {}
            logger.info("Using automatic device selection (SentenceTransformer will use available GPUs)")
        elif device.startswith("cuda"):
            model_kwargs = {"device": device}
            logger.info(f"Using device: {device}")
        elif device == "cpu":
            model_kwargs = {"device": "cpu"}
            logger.info("Using CPU")
        else:
            model_kwargs = {"device": device}
            logger.info(f"Using device: {device}")
            
        encode_kwargs = {"normalize_embeddings": True}
        
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        
        model_init_time = time.time() - model_start_time
        logger.info(f"Embedding model initialized in {model_init_time:.2f} seconds")
        
        # Create vector store
        logger.info("Creating FAISS vector store from documents...")
        vectorstore_start_time = time.time()
        
        vectorstore = FAISS.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
        
        vectorstore_time = time.time() - vectorstore_start_time
        logger.info(f"Vector store created in {vectorstore_time:.2f} seconds")
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        # Generate output path with naming convention
        model_name = embedding_model_name.split('/')[-1]
        db_name = f"{source_name}_{splitter_type}_{model_name}_cs{chunk_size}_co{chunk_overlap}"
        db_path = output_dir / db_name
        
        logger.info(f"Saving vector store to: {db_path}")
        save_start_time = time.time()
        
        # Save vector store
        vectorstore.save_local(db_path)
        
        save_time = time.time() - save_start_time
        logger.info(f"Vector store saved in {save_time:.2f} seconds")
        
        # Create ragger.info metadata file
        logger.info("Creating ragger.info metadata file")
        ragger_info = {
            "created_at": datetime.now().isoformat(),
            "embedding_model": embedding_model_name,
            "source_path": str(source_name),
            "splitter_type": splitter_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "total_documents": source_document_count or 0,
            "total_chunks": len(chunks),
            "device": device,
            "creation_time_seconds": time.time() - start_time,
            "vector_store_type": "FAISS",
            "ragger_version": "0.1.0"
        }
        
        info_path = db_path / "ragger.info"
        with open(info_path, 'w') as f:
            json.dump(ragger_info, f, indent=2)
        
        logger.info(f"Metadata saved to: {info_path}")
        
    except Exception as e:
        logger.error(f"Error creating vectorstore: {str(e)}")
        logger.exception("Full traceback:")
        raise
    
    total_time = time.time() - start_time
    logger.info(f"Vectorstore creation completed in {total_time:.2f} seconds")
    logger.info(f"Final vector store location: {db_path}")
    
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
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Enable verbose logging (DEBUG level and third-party library logs)")
    parser.add_argument("--device", default="auto", 
                      help="Device mapping: 'auto' for multi-GPU, 'cuda:0' for single GPU, 'cpu' for CPU (default: auto)")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("=== Markdown to Vector Database Creation Started ===")
    logger.info(f"Source: {args.source}")
    logger.info(f"Splitter: {args.splitter}")
    logger.info(f"Embedding model: {args.embedding_model}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Chunk size: {args.chunk_size}")
    logger.info(f"Chunk overlap: {args.chunk_overlap}")
    logger.info(f"Verbose mode: {args.verbose}")
    
    try:
        # Determine source name (for db naming)
        source_path = Path(args.source)
        source_name = source_path.stem if source_path.is_file() else source_path.name
        
        # Process the documents
        documents = load_documents(args.source)
        if not documents:
            logger.warning(f"No markdown files found in {args.source}")
            return
        
        chunks = split_documents(
            documents, 
            args.splitter, 
            args.chunk_size,
            args.chunk_overlap
        )
        
        if not chunks:
            logger.error("No chunks were created from the documents")
            return
        
        db_path = create_vectorstore(
            chunks, 
            args.embedding_model, 
            source_name, 
            args.splitter, 
            args.output_dir,
            args.device,
            args.chunk_size,
            args.chunk_overlap,
            len(documents)
        )
        
        logger.info("=== Vector database creation completed successfully! ===")
        logger.info(f"Final results:")
        logger.info(f"  Source: {args.source}")
        logger.info(f"  Documents processed: {len(documents)}")
        logger.info(f"  Chunks created: {len(chunks)}")
        logger.info(f"  Splitter: {args.splitter}")
        logger.info(f"  Embedding model: {args.embedding_model}")
        logger.info(f"  Vector database: {db_path}")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user (SIGINT)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()