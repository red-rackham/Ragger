#!/usr/bin/env python3
"""
directory_to_docling_json.py

Process an entire directory of documents (PDF, DOCX, PPTX, etc.) into docling JSON with annotations.
Uses docling's native batch processing capabilities to handle mixed file formats and
adds image descriptions using Ollama.
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


# Custom serializer only for markdown export
class DescriptionPictureSerializer(MarkdownPictureSerializer):
    """
    Custom picture serializer that replaces image references with their descriptions for markdown export.
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


def process_directory(input_dir, output_dir, export_markdown=False, image_scale=1.0, recursive=False):
    """
    Process all supported documents in a directory and convert them to docling JSON files.
    
    Args:
        input_dir: Path to the input directory
        output_dir: Path to save outputs
        export_markdown: Whether to export markdown with descriptions
        image_scale: Scale factor for image resolution (1.0 = original size, 0.5 = half size, etc.)
        recursive: Whether to process subdirectories
    
    Returns:
        tuple: (Number of successful conversions, number of failed conversions)
    """
    logger.info(f"Processing documents in directory: {input_dir}")

    # Configure pipeline options with image description
    pipeline_options = PdfPipelineOptions()
    pipeline_options.enable_remote_services = True
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = PictureDescriptionApiOptions(
        url=OLLAMA_ENDPOINT,
        params=dict(model=OLLAMA_MODEL),
        prompt=DESCRIPTION_PROMPT,
        timeout=90,
        response_format=ResponseFormat.MARKDOWN,
    )

    # Configure image extraction settings
    pipeline_options.images_scale = image_scale
    pipeline_options.generate_picture_images = True

    # Create document converter with options for PDF
    # The converter will handle other formats with default settings
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )

    # Find all files to process
    pattern = "**/*.*" if recursive else "*.*"
    all_files = list(Path(input_dir).glob(pattern))

    # Filter to only include supported file types
    supported_extensions = [
        ".pdf", ".docx", ".pptx", ".xlsx", ".html",
        ".md", ".csv", ".xml", ".asciidoc", ".adoc",
        ".jpg", ".jpeg", ".png", ".bmp", ".tiff"
    ]
    input_files = [f for f in all_files if f.is_file() and f.suffix.lower() in supported_extensions]

    if not input_files:
        logger.warning(f"No supported files found in {input_dir}")
        return 0, 0

    logger.info(f"Found {len(input_files)} files to process")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Start timing the total processing
    start_time = time.time()

    # Batch process all files using docling's native convert_all method
    results = doc_converter.convert_all(input_files, raises_on_error=False)

    # Process results
    successful = 0
    failed = 0

    for result in results:
        if result.status == "success" and result.document:
            try:
                # Generate output filename base
                doc_filename = result.input.file.stem

                # Export the document to JSON (primary output)
                # Note: All annotations (including image descriptions) are natively included
                # in the JSON format without needing any special serialization
                json_path = Path(output_dir) / f"{doc_filename}.json"
                result.document.save_as_json(json_path)
                logger.info(f"Document JSON exported to: {json_path}")

                # Export markdown if requested (using custom serializer to replace
                # images with their text descriptions)
                if export_markdown:
                    markdown_path = Path(output_dir) / f"{doc_filename}.md"
                    result.document.save_as_markdown(
                        markdown_path,
                        image_mode=ImageRefMode.PLACEHOLDER,
                        picture_serializer=DescriptionPictureSerializer()
                    )
                    logger.info(f"Markdown exported to: {markdown_path}")

                successful += 1
            except Exception as e:
                logger.error(f"Error exporting {result.input.file}: {e}")
                failed += 1
        else:
            logger.error(f"Failed to process {result.input.file}: {result.error}")
            failed += 1

    # Calculate processing time
    processing_time = time.time() - start_time
    logger.info(f"Processed {len(input_files)} files in {processing_time:.2f} seconds")

    return successful, failed


def main():
    """Process documents in input directory and save results to output directory."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process documents (PDF, DOCX, PPTX, etc.) into docling JSON with annotations"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing documents to process"
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
        help="Process files in subdirectories recursively"
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

    # Start timing the total processing
    total_time = time.time()

    # Process all documents in the directory
    successful, failed = process_directory(
        input_dir,
        output_dir,
        args.markdown,
        args.image_scale,
        args.recursive
    )

    # Print summary
    total_time_end = time.time()
    total_processing_time = total_time_end - total_time

    # Print detailed summary
    logger.info("=" * 60)
    logger.info(f"Processing complete. Successfully processed {successful} documents.")

    if failed > 0:
        logger.warning(f"Failed to process {failed} documents. Check the logs for details.")

    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total processing time: {total_processing_time:.2f} seconds")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
