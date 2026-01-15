"""
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[Path] = None,
    format_string: str = None,
):
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ProgressLogger:
    """Logger that tracks and displays progress."""

    def __init__(self, name: str, total: int = None):
        self.logger = get_logger(name)
        self.total = total
        self.current = 0

    def update(self, n: int = 1, message: str = None):
        """Update progress."""
        self.current += n
        if message:
            self.logger.info(message)

    def log(self, message: str, level: str = 'info'):
        """Log a message."""
        getattr(self.logger, level.lower())(message)

    def progress_string(self) -> str:
        """Get progress string."""
        if self.total:
            pct = 100 * self.current / self.total
            return f"[{self.current}/{self.total}] {pct:.1f}%"
        return f"[{self.current}]"
