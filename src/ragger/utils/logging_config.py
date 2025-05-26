"""
Centralized logging configuration for Ragger.
Ensures all logging goes to files and never to stdout/stderr to avoid breaking the CLI UI.
"""
import logging
from pathlib import Path


def setup_logging(verbose: bool = False):
    """Setup centralized logging for the entire application.
    
    Args:
        verbose: Enable debug level logging if True
    """
    # Use ~/.ragger/logs directory for application logs
    logs_dir = Path.home() / ".ragger" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear any existing handlers to avoid stdout/stderr leakage
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set logging level
    level = logging.DEBUG if verbose else logging.INFO
    
    # Create file handler
    log_file = logs_dir / "ragger.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    logging.root.setLevel(level)
    logging.root.handlers = [file_handler]
    
    # Ensure specific loggers also only use file handlers
    for logger_name in ['ragger', 'httpcore', 'httpx', 'asyncio']:
        logger = logging.getLogger(logger_name)
        logger.handlers = [file_handler]
        logger.propagate = False  # Don't propagate to avoid duplicate logs


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance that's guaranteed to only log to files.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance configured for file-only output
    """
    logger = logging.getLogger(name)
    
    # Ensure this logger doesn't accidentally write to stdout/stderr
    logger.propagate = False
    
    # If no handlers are set, add the file handler
    if not logger.handlers:
        logs_dir = Path.home() / ".ragger" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "ragger.log"
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger