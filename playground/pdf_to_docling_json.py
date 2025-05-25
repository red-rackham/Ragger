#!/usr/bin/env python3
"""
pdf_to_docling_json.py

Transform PDFs into docling JSON with annotations.
This script processes a folder of PDFs and generates JSON files with annotations,
particularly image descriptions using Ollama. All annotations are natively stored
in the docling JSON format without any custom serialization needed.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import DoclingDocument, PictureItem, PictureDescriptionData
from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import MarkdownDocSerializer, MarkdownPictureSerializer

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    ResponseFormat,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configuration for Ollama
OLLAMA_MODEL = "granite3.2-vision:2b"  # Same model as in other playground scripts
OLLAMA_ENDPOINT = "http://localhost:11434/v1/chat/completions"
DESCRIPTION_PROMPT = "Describe this image in detail for a visually impaired person."

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Custom serializer only for markdown export - not needed for JSON
# as annotations are natively stored in the docling JSON format
class DescriptionPictureSerializer(MarkdownPictureSerializer):
    """
    Custom picture serializer that replaces image references with their descriptions.
    Only used for markdown export, not needed for JSON export.
    """
    def serialize(self, *, item: PictureItem, doc_serializer: BaseDocSerializer,
                 doc: DoclingDocument, **kwargs) -> SerializationResult:
        # Look for picture descriptions in annotations
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


def process_pdf(pdf_path, output_dir, export_markdown=False, image_scale=1.0):
    """
    Process a single PDF file and save outputs to the given directory.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the output files
        export_markdown: Whether to export markdown (optional, default False)
        image_scale: Scale factor for image resolution (default 1.0)
        
    Returns:
        bool: True if processing was successful, False otherwise
    """
    logger.info(f"Processing document: {pdf_path}")

    # Configure the standard PDF Pipeline options
    pipeline_options = PdfPipelineOptions()

    # Enable remote services (required for API-based models like Ollama)
    pipeline_options.enable_remote_services = True

    # Enable picture description as a separate enrichment
    pipeline_options.do_picture_description = True

    # Configure Ollama for picture descriptions
    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        url=OLLAMA_ENDPOINT,
        params=dict(
            model=OLLAMA_MODEL,
        ),
        prompt=DESCRIPTION_PROMPT,
        timeout=90,
        response_format=ResponseFormat.MARKDOWN,
    )

    # Enable image extraction and set resolution
    pipeline_options.images_scale = image_scale
    pipeline_options.generate_picture_images = True

    # Create the document converter with standard PDF pipeline
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                # No pipeline_cls specified means it uses the standard PDF pipeline
            )
        }
    )

    try:
        # Process the document
        start_time = time.time()
        conversion_result = doc_converter.convert(pdf_path)
        doc = conversion_result.document
        processing_time = time.time() - start_time

        # Log useful document statistics
        num_pages = doc.num_pages()
        num_pictures = len(doc.pictures)
        num_tables = len(doc.tables)
        num_text_items = len(doc.texts)

        logger.info(f"Document processed in {processing_time:.2f} seconds")
        logger.info(f"Document statistics:")
        logger.info(f"  - Pages: {num_pages}")
        logger.info(f"  - Pictures: {num_pictures}")
        logger.info(f"  - Tables: {num_tables}")
        logger.info(f"  - Text items: {num_text_items}")

        # Generate output filename base
        doc_filename = conversion_result.input.file.stem

        # Export the document to JSON (primary output)
    # Note: All annotations (including image descriptions) are natively included
    # in the JSON format without needing any special serialization
        json_path = output_dir / f"{doc_filename}.json"
        doc.save_as_json(json_path)
        logger.info(f"Document JSON exported to: {json_path}")

        # Export markdown if requested (using custom serializer to replace
        # images with their text descriptions)
        if export_markdown:
            markdown_path = output_dir / f"{doc_filename}.md"
            doc.save_as_markdown(
                markdown_path,
                image_mode=ImageRefMode.PLACEHOLDER,
                picture_serializer=DescriptionPictureSerializer()
            )
            logger.info(f"Markdown exported to: {markdown_path}")

        return True
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {e}")
        return False


def main():
    """Process PDFs in input directory and save results to output directory."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Transform PDFs into docling JSON with annotations"
    )
    parser.add_argument(
        "input_dir", 
        help="Directory containing PDF files to process"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for processed files (default: input_dir/docling_output)"
    )
    parser.add_argument(
        "--markdown", "-m",
        action="store_true",
        help="Export markdown files with image descriptions (optional)"
    )
    parser.add_argument(
        "--image-scale", "-s",
        type=float,
        default=1.0,
        help="Scale factor for image resolution (default: 1.0)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process PDF files in subdirectories recursively"
    )
    
    args = parser.parse_args()

    # Set up input and output paths
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        logger.error(f"Input directory does not exist or is not a directory: {input_dir}")
        sys.exit(1)

    # If output directory not specified, create a 'docling_output' folder in the input directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / "docling_output"

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using output directory: {output_dir}")

    # Find all PDF files in the input directory
    pattern = "**/*.pdf" if args.recursive else "*.pdf"
    pdf_files = list(input_dir.glob(pattern))
    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        sys.exit(0)

    logger.info(f"Found {len(pdf_files)} PDF files to process")

    # Start timing the total processing
    total_time = time.time()

    # Process each PDF file
    successful = 0
    failed = 0

    for i, pdf_file in enumerate(pdf_files):
        logger.info(f"Processing ({i + 1}/{len(pdf_files)}): {pdf_file}")
        try:
            if process_pdf(pdf_file, output_dir, args.markdown, args.image_scale):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Unhandled error processing {pdf_file}: {e}")
            failed += 1

    # Print summary
    total_time_end = time.time()
    total_processing_time = total_time_end - total_time

    # Print detailed summary
    logger.info("=" * 60)
    logger.info(f"Processing complete. Successfully processed {successful} of {len(pdf_files)} PDFs.")

    if failed > 0:
        logger.warning(f"Failed to process {failed} PDFs. Check the logs for details.")

    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Processing time: {total_processing_time:.2f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()