import os
from pathlib import Path
import logging

ROOT = Path(os.getcwd())


def get_logger():
    """Initializes and returns a logger for the application."""

    logger = logging.getLogger(__name__)

    logger.setLevel(logging.INFO)

    # Create a console handler and set level to INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger
