"""Centralized logging configuration for the Video Profanity Censor."""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_path: Path) -> None:
    """Configure logging with a file handler and a stderr handler for errors.

    Args:
        log_path: Path to the log file.
    """
    root_logger = logging.getLogger("video_profanity_censor")
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.ERROR)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)
