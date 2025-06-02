#!/usr/bin/env python3
"""
PDF to JSON/HTML Multi-Method Converter

A PDF conversion testing script that uses docling to test various OCR methods,
vision models, and processing configurations. Supports output as JSON or HTML
split page view with embedded images and AI-generated descriptions.

Features:
- Multiple OCR engines (Tesseract, RapidOCR, EasyOCR)
- Vision model integration via Ollama for image descriptions
- GPU acceleration support
- Quality analysis and method comparison
- HTML split page output with embedded PDF page images
- Batch processing with detailed performance metrics

Usage:
    python pdf_to_json_multi_methods.py document.pdf
    python pdf_to_json_multi_methods.py document.pdf --output-format html_split_page
    python pdf_to_json_multi_methods.py document.pdf --methods standard tesseract_selective
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    EasyOcrOptions,
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
    RapidOcrOptions,
    ResponseFormat,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ollama service endpoints
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_ENDPOINT = f"{OLLAMA_BASE_URL}/v1/chat/completions"
OLLAMA_VERSION_ENDPOINT = f"{OLLAMA_BASE_URL}/api/version"
OLLAMA_MODELS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

# Vision processing settings
DESCRIPTION_PROMPT = "Describe this image in detail for a visually impaired person."
VISION_TIMEOUT = 150  # 2.5 minutes for vision model processing

# Test parameters
SCALES_TO_TEST = [1]
VISION_MODELS = [
    "granite3.2-vision:2b",
    # "llava:7b",
    # "llava:13b",
    # "minicpm-v:8b"

]

# Performance settings
USE_GPU = True
NUM_THREADS = 8

# Output format settings
DEFAULT_OUTPUT_FORMAT = "html_split_page"

# Method type constants
BASE_METHODS = [
    "standard",
    "tesseract_full_page",
    "tesseract_selective", 
    "rapidocr",
    "easyocr"
]

VISION_METHODS = [
    "standard_with_vision",
    "tesseract_selective_with_vision",
    "rapidocr_with_vision",
    "easyocr_with_vision", 
    "pypdfium2_with_vision"
]


# =============================================================================
# LOGGING SETUP
# =============================================================================

class CustomFormatter(logging.Formatter):
    """Custom formatter that only shows level for WARNING and above."""
    
    def format(self, record):
        if record.levelno >= logging.WARNING:
            return f"[{record.levelname}] {record.getMessage()}"
        else:
            return record.getMessage()


# Configure logging with custom formatter
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

class ConversionMethod:
    """Represents a PDF conversion method with its configuration."""
    
    def __init__(self, name: str, description: str, pipeline_options: PdfPipelineOptions,
                 backend=None, scale: float = 1.0, vision_model: str = None):
        self.name = name
        self.description = description
        self.pipeline_options = pipeline_options
        self.backend = backend
        self.scale = scale
        self.vision_model = vision_model


# =============================================================================
# CONFIGURATION FUNCTIONS
# =============================================================================

def create_base_pipeline_options() -> PdfPipelineOptions:
    """Create base pipeline options with common settings."""
    global USE_GPU, NUM_THREADS
    options = PdfPipelineOptions()
    
    # Enable advanced features
    options.do_table_structure = True
    options.do_formula_enrichment = True
    options.do_code_enrichment = True
    options.generate_picture_images = True
    options.generate_page_images = True
    
    # Configure GPU acceleration if available
    if USE_GPU:
        options.accelerator_options = AcceleratorOptions(
            num_threads=NUM_THREADS,
            device=AcceleratorDevice.AUTO
        )
    
    return options


def configure_tesseract_ocr(options: PdfPipelineOptions, force_full_page_ocr: bool, **kwargs):
    """Configure Tesseract OCR options for the given pipeline options."""
    options.do_ocr = True
    options.ocr_options = TesseractOcrOptions(
        force_full_page_ocr=force_full_page_ocr,
        **kwargs
    )


def configure_rapidocr(options: PdfPipelineOptions, force_full_page_ocr: bool, **kwargs):
    """Configure RapidOCR options for the given pipeline options."""
    options.do_ocr = True
    options.ocr_options = RapidOcrOptions(
        force_full_page_ocr=force_full_page_ocr,
        **kwargs
    )


def configure_easyocr(options: PdfPipelineOptions):
    """Configure EasyOCR options for the given pipeline options."""
    options.do_ocr = True
    options.ocr_options = EasyOcrOptions(
        lang=["en", "fr", "de", "es"],
        confidence_threshold=0.5,
        force_full_page_ocr=False
    )


def configure_vision_model(options: PdfPipelineOptions, vision_model: str):
    """Configure vision model options for the given pipeline options."""
    options.enable_remote_services = True
    options.do_picture_description = True
    options.picture_description_options = PictureDescriptionApiOptions(
        url=OLLAMA_ENDPOINT,
        params=dict(model=vision_model),
        prompt=DESCRIPTION_PROMPT,
        timeout=VISION_TIMEOUT,
        response_format=ResponseFormat.MARKDOWN,
    )


# =============================================================================
# OLLAMA SERVICE FUNCTIONS
# =============================================================================

def check_ollama_service() -> bool:
    """Check if Ollama service is available and responsive."""
    try:
        response = requests.get(OLLAMA_VERSION_ENDPOINT, timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def check_ollama_models(models: List[str]) -> List[str]:
    """Check which models are available in Ollama."""
    available_models = []
    try:
        response = requests.get(OLLAMA_MODELS_ENDPOINT, timeout=10)
        if response.status_code == 200:
            data = response.json()
            installed_models = [model["name"] for model in data.get("models", [])]
            available_models = [model for model in models if model in installed_models]
    except requests.exceptions.RequestException:
        pass
    return available_models


# =============================================================================
# METHOD CREATION
# =============================================================================

def create_conversion_methods() -> List[ConversionMethod]:
    """Create different conversion methods to test with all combinations of scales and vision models."""
    global SCALES_TO_TEST, VISION_MODELS
    methods = []
    
    # Mapping of method names to configuration functions
    config_funcs = {
        "standard": lambda opts: opts,
        "tesseract_full_page": lambda opts: configure_tesseract_ocr(opts, True, lang=["auto"]),
        "tesseract_selective": lambda opts: configure_tesseract_ocr(opts, False, lang=["auto"]),
        "rapidocr": lambda opts: configure_rapidocr(opts, True),
        "easyocr": configure_easyocr,
        "standard_with_vision": lambda opts, model: configure_vision_model(opts, model),
        "tesseract_selective_with_vision": lambda opts, model: (
            configure_tesseract_ocr(opts, False, lang=["auto"]),
            configure_vision_model(opts, model)
        ),
        "rapidocr_with_vision": lambda opts, model: (
            configure_rapidocr(opts, True),
            configure_vision_model(opts, model)
        ),
        "easyocr_with_vision": lambda opts, model: (
            configure_easyocr(opts),
            configure_vision_model(opts, model)
        ),
        "pypdfium2_with_vision": lambda opts, model: configure_vision_model(opts, model)
    }
    
    # Generate base methods for each scale
    for scale in SCALES_TO_TEST:
        for name in BASE_METHODS:
            options = create_base_pipeline_options()
            options.images_scale = scale
            config_funcs[name](options)
            
            methods.append(ConversionMethod(
                name=f"{name}_scale{scale}",
                description=f"{name.replace('_', ' ').capitalize()} conversion (scale {scale})",
                pipeline_options=options,
                scale=scale
            ))
    
    # Generate vision methods for each scale and vision model combination
    for scale in SCALES_TO_TEST:
        for vision_model in VISION_MODELS:
            for name in VISION_METHODS:
                options = create_base_pipeline_options()
                options.images_scale = scale
                
                # Handle PyPdfium2 backend separately
                backend = None
                if "pypdfium2" in name:
                    backend = PyPdfiumDocumentBackend
                
                config_funcs[name](options, vision_model)
                
                model_short = vision_model.split(':')[0].replace('granite3.2-vision', 'granite')
                methods.append(ConversionMethod(
                    name=f"{name}_{model_short}_scale{scale}",
                    description=f"{name.replace('_', ' ').capitalize()} conversion ({vision_model}, scale {scale})",
                    pipeline_options=options,
                    backend=backend,
                    scale=scale,
                    vision_model=vision_model
                ))
    
    return methods


# =============================================================================
# QUALITY ANALYSIS
# =============================================================================

def calculate_content_quality(doc, method: ConversionMethod) -> Dict:
    """Calculate content quality metrics for the conversion result."""
    quality = {}
    
    # Basic content metrics (totals)
    total_text_length = sum(len(text.text) for text in doc.texts if hasattr(text, 'text'))
    total_words = sum(len(text.text.split()) for text in doc.texts if hasattr(text, 'text'))
    
    quality["total_characters"] = total_text_length
    quality["total_words"] = total_words
    quality["content_density_score"] = total_text_length
    
    # Text quality indicators
    if total_text_length > 0:
        text_content = " ".join(text.text for text in doc.texts if hasattr(text, 'text'))
        
        # Simple quality heuristics
        quality["has_repeated_chars"] = any(char * 5 in text_content for char in "abcdefghijklmnopqrstuvwxyz")
        quality["special_char_ratio"] = round(sum(1 for c in text_content if not c.isalnum() and not c.isspace()) / len(text_content), 3)
        quality["whitespace_ratio"] = round(text_content.count(' ') / len(text_content), 3)
    
    # Structure detection totals
    quality["total_tables"] = len(doc.tables)
    quality["total_images"] = len(doc.pictures)
    quality["total_text_items"] = len(doc.texts)
    
    # Vision quality (if applicable)
    if method.vision_model:
        total_annotation_length = sum(len(" ".join(ann.text for ann in pic.annotations)) 
                                    for pic in doc.pictures)
        annotated_images = len([pic for pic in doc.pictures if pic.annotations])
        
        quality["total_annotation_chars"] = total_annotation_length
        quality["annotated_images_count"] = annotated_images
        quality["avg_annotation_length"] = round(total_annotation_length / len(doc.pictures), 1) if doc.pictures else 0
    
    # OCR method tracking
    if hasattr(method.pipeline_options, 'ocr_options') and method.pipeline_options.ocr_options:
        quality["ocr_method"] = type(method.pipeline_options.ocr_options).__name__
    else:
        quality["ocr_method"] = "Default" if method.pipeline_options.do_ocr else "None"
    
    return quality


def analyze_quality_across_methods(results: List[Dict]) -> Dict:
    """Analyze quality metrics across all successful methods."""
    successful_results = [r for r in results if r["success"] and "quality_metrics" in r["stats"]]
    if not successful_results:
        return {}
    
    analysis = {}
    
    # Content volume comparison
    word_counts = [r["stats"]["quality_metrics"]["total_words"] for r in successful_results]
    char_counts = [r["stats"]["quality_metrics"]["total_characters"] for r in successful_results]
    
    analysis["content_variation"] = {
        "max_words": max(word_counts),
        "min_words": min(word_counts),
        "word_count_range": max(word_counts) - min(word_counts),
        "avg_words": round(sum(word_counts) / len(word_counts), 1),
        "max_chars": max(char_counts),
        "min_chars": min(char_counts),
        "char_count_range": max(char_counts) - min(char_counts)
    }
    
    # Method consensus (methods that produce similar word counts)
    avg_words = analysis["content_variation"]["avg_words"]
    consensus_threshold = 0.1  # 10% variation
    consensus_methods = []
    outlier_methods = []
    
    for result in successful_results:
        words = result["stats"]["quality_metrics"]["total_words"]
        if abs(words - avg_words) / avg_words <= consensus_threshold:
            consensus_methods.append(result["method"])
        else:
            outlier_methods.append((result["method"], words))
    
    analysis["consensus"] = {
        "consensus_methods": consensus_methods,
        "outlier_methods": outlier_methods,
        "consensus_rate": round(len(consensus_methods) / len(successful_results), 2)
    }
    
    # Vision quality comparison (if applicable)
    vision_results = [r for r in successful_results if "total_annotation_chars" in r["stats"]["quality_metrics"]]
    if vision_results:
        annotation_lengths = [r["stats"]["quality_metrics"]["total_annotation_chars"] for r in vision_results]
        analysis["vision_quality"] = {
            "max_annotation_chars": max(annotation_lengths),
            "min_annotation_chars": min(annotation_lengths),
            "avg_annotation_chars": round(sum(annotation_lengths) / len(annotation_lengths), 1)
        }
    
    return analysis


# =============================================================================
# PDF PROCESSING
# =============================================================================

def process_pdf_with_method(pdf_path: Path, method: ConversionMethod, output_dir: Path, output_format: str = DEFAULT_OUTPUT_FORMAT) -> Dict:
    """Process a PDF with a specific conversion method."""
    method_start_time = time.time()
    
    logger.info(f"=" * 80)
    logger.info(f"STARTING METHOD: {method.name}")
    logger.info(f"Description: {method.description}")
    logger.info(f"Scale: {method.scale}")
    logger.info(f"Vision Model: {method.vision_model or 'None'}")
    logger.info(f"Backend: {method.backend.__name__ if method.backend else 'Default'}")
    
    # Log pipeline configuration details
    opts = method.pipeline_options
    logger.info(f"Pipeline Configuration:")
    logger.info(f"  - OCR enabled: {opts.do_ocr}")
    if hasattr(opts, 'ocr_options') and opts.ocr_options:
        if hasattr(opts.ocr_options, 'engine'):
            logger.info(f"  - OCR engine: {opts.ocr_options.engine}")
        if hasattr(opts.ocr_options, 'force_full_page_ocr'):
            logger.info(f"  - Full page OCR: {opts.ocr_options.force_full_page_ocr}")
    logger.info(f"  - Table structure: {opts.do_table_structure}")
    logger.info(f"  - Formula enrichment: {opts.do_formula_enrichment}")
    logger.info(f"  - Code enrichment: {opts.do_code_enrichment}")
    logger.info(f"  - Picture descriptions: {opts.do_picture_description}")
    logger.info(f"  - Generate picture images: {opts.generate_picture_images}")
    logger.info(f"  - Generate page images: {opts.generate_page_images}")
    logger.info(f"  - Image scale: {opts.images_scale}")
    if hasattr(opts, 'accelerator_options') and opts.accelerator_options:
        logger.info(f"  - GPU acceleration: {opts.accelerator_options.device}")
        logger.info(f"  - Threads: {opts.accelerator_options.num_threads}")
    if hasattr(opts, 'picture_description_options') and opts.picture_description_options:
        logger.info(f"  - Vision timeout: {opts.picture_description_options.timeout}s")
        logger.info(f"  - Vision prompt: {opts.picture_description_options.prompt}")
    logger.info(f"-" * 80)
    
    result = {
        "method": method.name,
        "description": method.description,
        "success": False,
        "processing_time": 0,
        "setup_time": 0,
        "conversion_time": 0,
        "save_time": 0,
        "error": None,
        "stats": {},
        "output_file": None
    }
    
    try:
        # Create converter with the specific method configuration
        setup_start = time.time()
        converter_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=method.pipeline_options
            )
        }
        
        # Use custom backend if specified
        if method.backend:
            converter_options[InputFormat.PDF].backend = method.backend
            logger.info(f"Using custom backend: {method.backend.__name__}")
            
        doc_converter = DocumentConverter(format_options=converter_options)
        setup_time = time.time() - setup_start
        logger.info(f"Converter setup completed in {setup_time:.2f}s")
        
        # Process the document
        logger.info(f"Starting document conversion...")
        conversion_start = time.time()
        conversion_result = doc_converter.convert(pdf_path)
        doc = conversion_result.document
        conversion_time = time.time() - conversion_start
        logger.info(f"Document conversion completed in {conversion_time:.2f}s")
        
        # Collect detailed statistics
        pages = doc.num_pages()
        pictures = len(doc.pictures)
        tables = len(doc.tables)
        text_items = len(doc.texts)
        
        logger.info(f"Document analysis:")
        logger.info(f"  - Pages: {pages}")
        logger.info(f"  - Pictures: {pictures}")
        logger.info(f"  - Tables: {tables}")
        logger.info(f"  - Text items: {text_items}")
        
        # Count annotations if any
        annotation_count = 0
        picture_with_annotations = 0
        for i, picture in enumerate(doc.pictures):
            pic_annotations = len(picture.annotations)
            annotation_count += pic_annotations
            if pic_annotations > 0:
                picture_with_annotations += 1
                # Print the actual annotation text
                for j, annotation in enumerate(picture.annotations):
                    if hasattr(annotation, 'text') and annotation.text:
                        logger.info(f"  - Picture {i+1} annotation {j+1}:")
                        logger.info(f"    \033[3m{annotation.text}\033[0m")

        if annotation_count > 0:
            logger.info(f"  - Picture annotations: {annotation_count} total, {picture_with_annotations} pictures with annotations")
        
        # Count other enrichments
        formula_count = sum(1 for item in doc.texts if hasattr(item, 'label') and 'formula' in str(item.label).lower())
        code_count = sum(1 for item in doc.texts if hasattr(item, 'label') and 'code' in str(item.label).lower())
        
        if formula_count > 0:
            logger.info(f"  - Formulas detected: {formula_count}")
        if code_count > 0:
            logger.info(f"  - Code blocks detected: {code_count}")
        
        # Calculate content quality metrics
        quality_metrics = calculate_content_quality(doc, method)
        
        stats = {
            "pages": pages,
            "pictures": pictures,
            "tables": tables,
            "text_items": text_items,
            "picture_annotations": annotation_count,
            "pictures_with_annotations": picture_with_annotations,
            "formulas": formula_count,
            "code_blocks": code_count,
            "scale": method.scale,
            "vision_model": method.vision_model,
            "setup_time_seconds": round(setup_time, 2),
            "conversion_time_seconds": round(conversion_time, 2),
            "quality_metrics": quality_metrics
        }
        
        # Save to JSON with descriptive filename
        logger.info(f"Saving results to {output_format}...")
        save_start = time.time()
        doc_filename = conversion_result.input.file.stem
        
        # Create filename with all parameters and scale
        filename_parts = [doc_filename, method.name, f"scale{method.scale}"]

        # Add vision model if present
        if method.vision_model:
            # Clean model name for filename
            model_clean = method.vision_model.replace(':', '-').replace('/', '-').replace('\\', '-').replace('.', '-')
            filename_parts.append(f"vision-{model_clean}")
        
        # Add backend if not default
        if method.backend:
            backend_name = method.backend.__name__.replace('DocumentBackend', '').replace('Backend', '')
            filename_parts.append(f"backend-{backend_name}")
        
        # Join with underscores and clean any remaining problematic characters
        descriptive_filename = "_".join(filename_parts).replace(' ', '-').replace('/', '-').replace('\\', '-')
        
        if output_format == "html_split_page":
            descriptive_filename += "_split_page.html"
            output_file = output_dir / descriptive_filename
            from docling_core.types.doc import ImageRefMode
            from docling_core.types.doc.labels import DocItemLabel
            from docling_core.types.doc.document import ContentLayer, PictureDescriptionData, PictureItem
            from docling_core.transforms.serializer.html import HTMLDocSerializer, HTMLPictureSerializer, HTMLOutputStyle
            from docling_core.transforms.serializer.base import BaseDocSerializer, SerializationResult
            from docling_core.transforms.serializer.common import create_ser_result
            from typing_extensions import override
            from typing import Any
            
            # Custom HTML picture serializer that includes descriptions
            class DescriptionHTMLPictureSerializer(HTMLPictureSerializer):
                @override
                def serialize(self, *, item: PictureItem, doc_serializer: BaseDocSerializer, doc: "DoclingDocument", **kwargs: Any) -> SerializationResult:
                    # Get the base HTML serialization (image)
                    base_result = super().serialize(item=item, doc_serializer=doc_serializer, doc=doc, **kwargs)
                    
                    # Look for picture description annotations
                    description = ""
                    for annotation in item.annotations:
                        if isinstance(annotation, PictureDescriptionData):
                            description = annotation.text
                            break
                    
                    # If we have a description, add it to the HTML
                    if description:
                        # Create description HTML with proper escaping
                        import html
                        escaped_desc = html.escape(description)
                        description_html = f'<div class="image-description"><strong>Description:</strong> {escaped_desc}</div>'
                        
                        # Combine image and description
                        combined_html = base_result.text
                        if combined_html:
                            # Remove the closing </figure> tag if present and add description inside
                            if combined_html.endswith("</figure>"):
                                combined_html = combined_html[:-9] + description_html + "</figure>"
                            else:
                                # If no figure wrapper, add description after the image
                                combined_html += description_html
                        else:
                            combined_html = description_html
                        
                        return create_ser_result(text=combined_html, span_source=item)
                    
                    # Return the base result if no description
                    return base_result
            
            # Create custom serializer with split page style and description support
            custom_serializer = HTMLDocSerializer(
                doc=doc,
                picture_serializer=DescriptionHTMLPictureSerializer()
            )
            custom_serializer.params.output_style = HTMLOutputStyle.SPLIT_PAGE
            custom_serializer.params.image_mode = ImageRefMode.EMBEDDED
            
            # Generate HTML content with descriptions
            html_content = custom_serializer.serialize().text
            
            # Write the HTML to file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
        else:  # default to json
            descriptive_filename += ".json"
            output_file = output_dir / descriptive_filename
            doc.save_as_json(output_file)
        
        save_time = time.time() - save_start
        
        if output_format == "html_split_page":
            logger.info(f"HTML split page saved in {save_time:.2f}s")
        else:
            logger.info(f"JSON saved in {save_time:.2f}s")
        logger.info(f"Output file: {descriptive_filename}")
        
        total_processing_time = time.time() - method_start_time
        
        result.update({
            "success": True,
            "processing_time": total_processing_time,
            "setup_time": setup_time,
            "conversion_time": conversion_time,
            "save_time": save_time,
            "stats": stats,
            "output_file": str(output_file)
        })
        
        logger.info(f"✓ METHOD COMPLETED: {method.name}")
        logger.info(f"  Total time: {total_processing_time:.2f}s (setup: {setup_time:.2f}s, conversion: {conversion_time:.2f}s, save: {save_time:.2f}s)")
        logger.info(f"  Content: {pages} pages, {pictures} pictures, {tables} tables, {annotation_count} annotations")
        logger.info(f"=" * 80)
        
    except Exception as e:
        total_processing_time = time.time() - method_start_time
        result.update({
            "error": str(e),
            "processing_time": total_processing_time
        })
        logger.error(f"✗ METHOD FAILED: {method.name}")
        logger.error(f"  Error: {e}")
        logger.error(f"  Failed after: {total_processing_time:.2f}s")
        logger.error(f"=" * 80)
    
    return result


# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_comparison_report(results: List[Dict], output_dir: Path, pdf_name: str, total_time: float):
    """Generate a comparison report of all methods."""
    successful_methods = len([r for r in results if r["success"]])
    report = {
        "pdf_file": pdf_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "methods_tested": len(results),
        "successful_methods": successful_methods,
        "success_rate": round(successful_methods / len(results) * 100, 1) if results else 0,
        "total_time_seconds": round(total_time, 2),
        "total_time_minutes": round(total_time / 60, 1),
        "average_time_per_method": round(total_time / len(results), 1) if results else 0,
        "results": results,
        "summary": {}
    }
    
    # Create summary statistics
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        processing_times = [r["processing_time"] for r in successful_results]
        report["summary"] = {
            "fastest_method": min(successful_results, key=lambda x: x["processing_time"])["method"],
            "slowest_method": max(successful_results, key=lambda x: x["processing_time"])["method"],
            "average_processing_time": round(sum(processing_times) / len(processing_times), 2),
            "methods_with_annotations": len([r for r in successful_results if r["stats"].get("picture_annotations", 0) > 0])
        }
    
    # Add quality analysis
    quality_analysis = analyze_quality_across_methods(results)
    if quality_analysis:
        report["quality_analysis"] = quality_analysis
    
    # Save report with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"{pdf_name}_comparison_report_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Comparison report saved to: {report_file}")
    
    # Clean, streamlined report
    print("\n" + "="*80)
    print(f"CONVERSION REPORT - {pdf_name}")
    print("="*80)
    
    # Summary stats
    if successful_results:
        fastest = min(successful_results, key=lambda x: x["processing_time"])
        slowest = max(successful_results, key=lambda x: x["processing_time"])
        
        print(f"Results: {report['successful_methods']}/{report['methods_tested']} successful ({report['success_rate']}%)")
        print(f"Runtime: {report['total_time_seconds']}s total, {report['average_time_per_method']}s avg")
        print(f"Range:   {fastest['method'].split('_')[0]} ({fastest['processing_time']:.1f}s) to {slowest['method'].split('_')[0]} ({slowest['processing_time']:.1f}s)")
    
    # Main results table
    print(f"\n{'Time':<8} {'Words':<8} {'Chars':<8} {'Tables':<7} {'Images':<7} {'Annot':<6} {'Method'}")
    print("-" * 100)
    
    for result in successful_results:
        method_name = result['method']
        time_val = result['processing_time']
        
        stats = result['stats']
        qm = stats.get('quality_metrics', {})
        words = qm.get('total_words', 0)
        chars = qm.get('total_characters', 0)
        tables = stats.get('tables', 0)
        images = stats.get('pictures', 0)
        annotations = stats.get('picture_annotations', 0)
        
        print(f"{time_val:<8.1f} {words:<8} {chars:<8} {tables:<7} {images:<7} {annotations:<6} {method_name}")
    
    # Show any failures
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        print(f"\nFailed Methods:")
        for result in failed_results:
            method_name = result['method']
            error_msg = result['error'][:50] + "..." if len(result['error']) > 50 else result['error']
            print(f"  {method_name}: {error_msg}")
    
    # Quality Analysis section
    if quality_analysis:
        print("\nQuality Analysis:")
        print("-"*80)
        
        if "content_variation" in quality_analysis:
            cv = quality_analysis["content_variation"]
            print(f"Content range: {cv['min_words']}-{cv['max_words']} words (avg: {cv['avg_words']})")
        
        if "consensus" in quality_analysis:
            consensus = quality_analysis["consensus"]
            print(f"Method consensus: {len(consensus['consensus_methods'])}/{len(successful_results)} methods agree")
            if consensus["outlier_methods"]:
                outlier_strs = [f"{method.split('_')[0]}({words}w)" for method, words in consensus["outlier_methods"]]
                print(f"Outliers: {', '.join(outlier_strs)}")
        
        if "vision_quality" in quality_analysis:
            vq = quality_analysis["vision_quality"]
            print(f"Vision annotations: {vq['min_annotation_chars']}-{vq['max_annotation_chars']} chars (avg: {vq['avg_annotation_chars']})")
    
    print("="*80)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Test various PDF conversion methods and compare results."""
    global USE_GPU, NUM_THREADS, SCALES_TO_TEST, VISION_MODELS
    
    parser = argparse.ArgumentParser(
        description="Test various PDF-to-JSON conversion methods with docling"
    )
    parser.add_argument(
        "pdf_file",
        help="PDF file to process with different methods"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for results (default: same directory as PDF)"
    )
    parser.add_argument(
        "--methods", "-m",
        nargs="+",
        help="Specific method prefixes to test (e.g., 'standard', 'tesseract_full_page')"
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=SCALES_TO_TEST,
        help=f"Scales to test (default: {SCALES_TO_TEST})"
    )
    parser.add_argument(
        "--vision-models",
        nargs="+",
        default=VISION_MODELS,
        help=f"Vision models to test (default: {VISION_MODELS})"
    )
    parser.add_argument(
        "--include-base", "-b",
        action="store_true",
        help="Include base methods without vision models (default: vision-only)"
    )
    parser.add_argument(
        "--skip-vision", "-s",
        action="store_true",
        help="Skip methods that require vision models (only test base methods)"
    )
    parser.add_argument(
        "--gpu", "--no-gpu",
        dest="gpu",
        action=argparse.BooleanOptionalAction,
        default=USE_GPU,
        help=f"Enable/disable GPU acceleration (default: {USE_GPU})"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=NUM_THREADS,
        help=f"Number of processing threads (default: {NUM_THREADS})"
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Show which methods would be tested without actually processing the PDF"
    )
    parser.add_argument(
        "--output-format", "-f",
        choices=["json", "html_split_page"],
        default=DEFAULT_OUTPUT_FORMAT,
        help=f"Output format for converted documents: json or html_split_page (default: {DEFAULT_OUTPUT_FORMAT})"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists() or pdf_path.suffix.lower() != '.pdf':
        logger.error(f"Invalid PDF file: {pdf_path}")
        return 1
    
    # Set up output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pdf_path.parent / f"{pdf_path.stem}_conversion_methods_output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Update global configuration from args
    USE_GPU = args.gpu
    NUM_THREADS = args.threads
    SCALES_TO_TEST = args.scales
    VISION_MODELS = args.vision_models
    
    # Print configuration summary
    print("\n" + "="*80)
    print("PDF TO JSON/HTML MULTI-METHODS CONVERTER - CONFIGURATION")
    print("="*80)
    print(f"Input PDF: {pdf_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {args.output_format}")
    print(f"Dry run mode: {'Yes' if args.dry_run else 'No'}")
    print()
    print("Processing Configuration:")
    print(f"  - GPU acceleration: {'Enabled' if USE_GPU else 'Disabled'}")
    print(f"  - Processing threads: {NUM_THREADS}")
    print(f"  - Vision timeout: {VISION_TIMEOUT}s")
    print(f"  - Vision prompt: '{DESCRIPTION_PROMPT}'")
    print()
    print("Pipeline Options (applied to all methods):")
    print(f"  - Table structure detection: {True}")
    print(f"  - Formula enrichment: {True}")
    print(f"  - Code enrichment: {True}")
    print(f"  - Generate picture images: {True}")
    print(f"  - Generate page images: {True}")
    print()
    print("Test Parameters:")
    print(f"  - Image scales: {SCALES_TO_TEST}")
    print(f"  - Vision models: {len(VISION_MODELS)} models")
    for i, model in enumerate(VISION_MODELS, 1):
        print(f"    {i}. {model}")
    print()
    print("Method Selection:")
    if args.skip_vision:
        print("  - Mode: Base methods only (--skip-vision)")
    elif args.include_base:
        print("  - Mode: Both base and vision methods (--include-base)")
    else:
        print("  - Mode: Vision-only methods (default)")
    
    if args.methods:
        print(f"  - Method filter: {', '.join(args.methods)}")
    else:
        print("  - Method filter: All methods")
    print("="*80)
    
    logger.info(f"Configuration loaded - GPU: {USE_GPU}, Threads: {NUM_THREADS}")
    
    # Check Ollama service if vision methods will be used
    if not args.skip_vision:
        print(f"\nChecking Ollama service availability...")
        if check_ollama_service():
            print("✓ Ollama service is running")
            available_models = check_ollama_models(VISION_MODELS)
            if available_models:
                print(f"✓ Available vision models: {', '.join(available_models)}")
                # Update VISION_MODELS to only include available models
                VISION_MODELS = available_models
            else:
                print(f"Warning: None of the configured vision models are installed in Ollama")
                print(f"   Configured models: {', '.join(VISION_MODELS)}")
                print(f"   Consider installing models with: ollama pull <model_name>")
        else:
            print("✗ Ollama service is not running or not accessible")
            print("   Vision methods will fail. Start Ollama service or use --skip-vision flag")
        print()
    
    # Get conversion methods with updated configuration
    all_methods = create_conversion_methods()
    
    # Filter methods if requested
    if args.methods:
        # Filter by method prefix
        methods_to_test = []
        for method in all_methods:
            for prefix in args.methods:
                if method.name.startswith(prefix):
                    methods_to_test.append(method)
                    break
        
        if not methods_to_test:
            available_prefixes = set(m.name.split('_')[0] for m in all_methods)
            logger.error(f"No methods found for prefixes: {args.methods}")
            logger.info(f"Available prefixes: {', '.join(sorted(available_prefixes))}")
            return 1
    else:
        methods_to_test = all_methods
    
    # Filter methods based on flags
    if args.skip_vision and args.include_base:
        logger.error("Cannot use both --skip-vision and --include-base flags together")
        return 1
    
    if args.skip_vision:
        # Only base methods, no vision
        methods_to_test = [m for m in methods_to_test if m.vision_model is None]
        logger.info("Testing only base methods (no vision models)")
    elif not args.include_base:
        # Default: only vision methods
        methods_to_test = [m for m in methods_to_test if m.vision_model is not None]
        logger.info("Testing only vision-enabled methods (default behavior)")
    else:
        # Include both base and vision methods
        logger.info("Testing both base and vision-enabled methods")
    
    if not methods_to_test:
        logger.error("No methods to test")
        return 1
    
    # Handle dry run mode
    if args.dry_run:
        print("\n" + "="*80)
        print(f"DRY RUN - Methods that would be tested for: {pdf_path.name}")
        print("="*80)
        print(f"Total methods: {len(methods_to_test)}")
        print(f"Output directory: {output_dir}")
        print("\nBase methods (no vision):")
        base_methods = [m for m in methods_to_test if m.vision_model is None]
        for i, method in enumerate(base_methods, 1):
            print(f"  {i:2d}. {method.name:<30} - {method.description}")
        
        vision_methods = [m for m in methods_to_test if m.vision_model is not None]
        if vision_methods:
            print("\nVision-enabled methods:")
            for i, method in enumerate(vision_methods, len(base_methods) + 1):
                print(f"  {i:2d}. {method.name:<30} - {method.description}")
        
        print("\nMethod breakdown by type:")
        method_counts = {}
        for method in methods_to_test:
            base_name = method.name.split('_scale')[0].split('_granite')[0].split('_llava')[0].split('_minicpm')[0]
            method_counts[base_name] = method_counts.get(base_name, 0) + 1
        
        for method_type, count in sorted(method_counts.items()):
            print(f"  - {method_type}: {count} variants")
        
        if vision_methods:
            print("\nVision models used:")
            vision_counts = {}
            for method in vision_methods:
                if method.vision_model:
                    vision_counts[method.vision_model] = vision_counts.get(method.vision_model, 0) + 1
            for model, count in sorted(vision_counts.items()):
                print(f"  - {model}: {count} methods")
        
        print("\nScales tested:")
        for scale in sorted(set(m.scale for m in methods_to_test)):
            count = len([m for m in methods_to_test if m.scale == scale])
            print(f"  - Scale {scale}: {count} methods")
        
        print("="*80)
        print("Use --help to see filtering options or remove --dry-run to execute.")
        return 0
    
    # Log test setup
    logger.info(f"=" * 80)
    logger.info(f"PDF CONVERSION TEST")
    logger.info(f"=" * 80)
    logger.info(f"Input file: {pdf_path}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output format: {args.output_format}")
    logger.info(f"Total methods to test: {len(methods_to_test)}")
    logger.info(f"Global configuration:")
    logger.info(f"  - GPU enabled: {USE_GPU}")
    logger.info(f"  - Threads: {NUM_THREADS}")
    logger.info(f"  - Vision timeout: {VISION_TIMEOUT}s")
    logger.info(f"  - Scales: {SCALES_TO_TEST}")
    logger.info(f"  - Vision models: {VISION_MODELS}")
    
    # Count method types
    base_methods = [m for m in methods_to_test if m.vision_model is None]
    vision_methods = [m for m in methods_to_test if m.vision_model is not None]
    
    logger.info(f"Method breakdown:")
    logger.info(f"  - Base methods (no vision): {len(base_methods)}")
    logger.info(f"  - Vision-enabled methods: {len(vision_methods)}")
    
    if vision_methods:
        vision_counts = {}
        for m in vision_methods:
            vision_counts[m.vision_model] = vision_counts.get(m.vision_model, 0) + 1
        for model, count in vision_counts.items():
            logger.info(f"    - {model}: {count} methods")
    
    logger.info(f"=" * 80)
    
    # Test each method
    results = []
    total_start_time = time.time()
    successful = 0
    failed = 0
    
    for i, method in enumerate(methods_to_test, 1):
        logger.info(f"{'#' * 80}")
        logger.info(f"Progress: {i}/{len(methods_to_test)} ({i/len(methods_to_test)*100:.1f}%)")

        result = process_pdf_with_method(pdf_path, method, output_dir, args.output_format)
        results.append(result)
        
        if result["success"]:
            successful += 1
            logger.info(f"✓ SUCCESS: {successful}/{len(methods_to_test)} methods completed successfully")
        else:
            failed += 1
            logger.info(f"✗ FAILURE: {failed}/{len(methods_to_test)} methods failed")
        
        # Show ETA if more than 1 method completed and methods still remaining
        if 1 < i < len(methods_to_test):
            elapsed = time.time() - total_start_time
            avg_time = elapsed / i
            remaining = (len(methods_to_test) - i) * avg_time
            eta_minutes = remaining / 60
            logger.info(f"ETA: {eta_minutes:.1f} minutes remaining ({avg_time:.1f}s avg per method)")
    
    total_time = time.time() - total_start_time
    
    # Generate comparison report
    generate_comparison_report(results, output_dir, pdf_path.stem, total_time)
    
    logger.info(f"TESTING COMPLETED!")
    logger.info(f"--> Comparison report and all outputs saved to: {output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())