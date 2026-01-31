"""
Logger configuration module for Planuze.

This module provides a setup function to configure logging with both console
and file handlers, ensuring consistent logging across the application.
"""
import logging
import os
import sys


def setup_logger(
    name: str = "planuze", log_level: str = "INFO"
) -> logging.Logger:
    """
    Configures and returns a logger with console and file handlers.

    Args:
        name: Name of the logger.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logging.Logger: Configured logger instance.
    """
    configured_logger = logging.getLogger(name)
    configured_logger.setLevel(log_level)

    # Prevent adding handlers multiple times
    if configured_logger.hasHandlers():
        return configured_logger

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    configured_logger.addHandler(console_handler)

    # File Handler
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(
        log_dir, "planuze.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    configured_logger.addHandler(file_handler)

    return configured_logger


# Default logger instance
logger = setup_logger(log_level=os.getenv("LOG_LEVEL", "INFO"))
