"""
Logging utilities for DepthSplat Inference Pipeline.

This module provides logging setup and configuration utilities.

Usage:
    from utils.logging_utils import setup_logging, get_logger
    
    setup_logging(level="INFO", use_colors=True)
    logger = get_logger(__name__)
    logger.info("Pipeline started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union


def setup_logging(
    level: Union[str, int] = "INFO",
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    file: Optional[str] = None,
    use_colors: bool = True,
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log message format
        file: Optional path to log file
        use_colors: Whether to use colored output (requires coloredlogs)
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format)
    
    # Try to use coloredlogs if available and requested
    if use_colors:
        try:
            import coloredlogs
            coloredlogs.install(
                level=level,
                fmt=format,
                logger=root_logger,
            )
            colors_installed = True
        except ImportError:
            colors_installed = False
    else:
        colors_installed = False
    
    # Add console handler if coloredlogs not installed
    if not colors_installed:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if file:
        file_path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging to file: {file}")


def setup_logging_from_config(config: Dict[str, Any]) -> None:
    """
    Setup logging from a configuration dictionary.
    
    Expected config format:
        {
            "level": "INFO",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "file": null,
            "use_colors": true
        }
    """
    setup_logging(
        level=config.get("level", "INFO"),
        format=config.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
        file=config.get("file"),
        use_colors=config.get("use_colors", True),
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggingContext:
    """
    Context manager for temporary logging level change.
    
    Usage:
        with LoggingContext(level=logging.DEBUG):
            # Verbose logging here
            pass
    """
    
    def __init__(self, level: int, logger_name: Optional[str] = None):
        self.level = level
        self.logger_name = logger_name
        self.old_level = None
    
    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.old_level = logger.level
        logger.setLevel(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.old_level)
        return False


def log_performance(func):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_performance
        def my_function():
            pass
    """
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            logger.debug(f"{func.__name__} completed in {elapsed:.2f}ms")
            return result
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"{func.__name__} failed after {elapsed:.2f}ms: {e}")
            raise
    
    return wrapper


if __name__ == "__main__":
    # Test logging setup
    setup_logging(level="DEBUG", use_colors=True)
    
    logger = get_logger(__name__)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    
    @log_performance
    def test_function():
        import time
        time.sleep(0.1)
        return "done"
    
    result = test_function()
    print(f"Result: {result}")
