#!/usr/bin/env python
"""
JSON to Markdown Converter with Image Descriptions

This script:
1. Loads docling JSON files from an input path (directory or single file)
2. Creates markdown files with image descriptions included
3. Saves the markdown files to an output directory

Usage:
    python json_to_md_with_descriptions.py input_path [output_dir]

    input_path: Path to a single docling JSON file or a directory containing JSON files
    output_dir: Optional output directory for the markdown files
    
    If output_dir is not specified, an 'md_output' folder will be 
    created in the same directory as the input.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Optional

from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownPictureSerializer
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DoclingDocument, PictureDescriptionData, PictureItem
from typing_extensions import override

logger = logging.getLogger(__name__)


class DescriptionPictureSerializer(MarkdownPictureSerializer):
    """
    Custom picture serializer that replaces image references with their descriptions.
    
    This serializer looks for picture descriptions in the annotations and returns
    the description text instead of an image reference. If no description is available,
    it falls back to the image caption or a default placeholder.
    """
    
    @override
    def serialize(
        self, 
        *, 
        item: PictureItem, 
        doc_serializer: BaseDocSerializer,
        doc: DoclingDocument, 
        **kwargs: Any
    ) -> SerializationResult:
        """
        Serialize a picture item to markdown, replacing the image with its description.
        
        Args:
            item: The picture item to serialize
            doc_serializer: The parent document serializer
            doc: The document containing the picture
            **kwargs: Additional keyword arguments
            
        Returns:
            A serialization result containing the description text
        """
        # Look for picture description annotations
        description = ""
        for annotation in item.annotations:
            if isinstance(annotation, PictureDescriptionData):
                description = annotation.text
                break
        
        # If no description is available, use the caption or a default placeholder
        if not description and item.caption_text(doc=doc):
            description = f"\n\n**Image Caption:** {item.caption_text(doc=doc)}\n\n"
        elif not description:
            description = "\n\n[Image without description]\n\n"
        else:
            description = f"\n\n**Image Description:** {description}\n\n"
        
        # Return the description instead of an image placeholder
        return create_ser_result(text=description, span_source=item)


def process_json_file(json_path: Path, output_dir: Path) -> bool:
    """
    Process a single JSON file and save markdown output to the given directory.
    
    Args:
        json_path: Path to the docling JSON file
        output_dir: Directory where the markdown file will be saved
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        logger.info(f"Processing document: {json_path}")
        
        # Load the document from JSON
        start_time = time.time()
        doc = DoclingDocument.load_from_json(json_path)
        
        # Log document statistics
        logger.info(f"Document statistics:")
        logger.info(f"  - Pages: {doc.num_pages()}")
        logger.info(f"  - Pictures: {len(doc.pictures)}")
        logger.info(f"  - Tables: {len(doc.tables)}")
        logger.info(f"  - Text items: {len(doc.texts)}")
        
        # Generate output filename
        markdown_path = output_dir / f"{json_path.stem}.md"
        
        # Create markdown serializer with our custom picture serializer
        serializer = MarkdownDocSerializer(
            doc=doc,
            picture_serializer=DescriptionPictureSerializer()
        )
        
        # Export markdown using the serializer
        markdown_content = serializer.serialize().text
        
        # Write markdown to file
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        processing_time = time.time() - start_time
        logger.info(f"Markdown exported to: {markdown_path} in {processing_time:.2f} seconds")
        
        return True
    except Exception as e:
        logger.error(f"Error processing {json_path}: {e}")
        return False


def main() -> int:
    """
    Process JSON files in input directory and save MD results to output directory.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Convert docling JSON files to markdown with image descriptions")
    parser.add_argument("input_path", help="Path to a docling JSON file or directory containing JSON files")
    parser.add_argument("output_dir", nargs="?", help="Output directory for markdown files (optional)")
    args = parser.parse_args()
    
    # Set up input path
    input_path = Path(args.input_path)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    # Determine if input is a file or directory
    json_files = []
    if input_path.is_file():
        if input_path.suffix.lower() == '.json':
            json_files = [input_path]
            parent_dir = input_path.parent
        else:
            logger.error(f"Input file is not a JSON file: {input_path}")
            return 1
    elif input_path.is_dir():
        # Find all .json files (case insensitive)
        json_files = list(input_path.glob("**/*.json"))
        json_files.extend(list(input_path.glob("**/*.JSON")))
        parent_dir = input_path
    else:
        logger.error(f"Input path is neither a file nor a directory: {input_path}")
        return 1
    
    # If output directory not specified, create an 'md_output' folder alongside the input
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = parent_dir / "md_output"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")
    if not json_files:
        logger.warning(f"No JSON files found in {input_path}")
        return 0
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Process each JSON file
    start_time = time.time()
    successful = 0
    failed = 0
    
    for i, json_file in enumerate(json_files, 1):
        logger.info(f"Processing ({i}/{len(json_files)}): {json_file}")
        if process_json_file(json_file, output_dir):
            successful += 1
        else:
            failed += 1
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    logger.info("=" * 60)
    logger.info(f"Processing complete. Successfully processed {successful} of {len(json_files)} JSON files.")
    
    if failed > 0:
        logger.warning(f"Failed to process {failed} JSON files. Check the logs for details.")
    
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total processing time: {processing_time:.2f} seconds")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())