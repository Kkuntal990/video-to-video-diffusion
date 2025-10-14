"""
Logging utilities
"""

import logging
import sys
from pathlib import Path


def setup_logger(name='video_diffusion', log_file=None, level=logging.INFO):
    """
    Setup logger for training/inference

    Args:
        name: logger name
        log_file: path to log file (optional)
        level: logging level

    Returns:
        logger: configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Test logger
    logger = setup_logger('test_logger', log_file='test.log')

    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")

    print("Logger test successful!")
