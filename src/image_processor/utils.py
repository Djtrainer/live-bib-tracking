import os
from pathlib import Path
import logging

ROOT = Path(os.getcwd())


def get_logger():
    """Initializes and returns a logger for the application."""

    logger = logging.getLogger(__name__)

    # Set logger level to DEBUG to pick up both info and debug messages
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    return logger
