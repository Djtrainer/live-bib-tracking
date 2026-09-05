"""Race CV: standalone finish-line detection and bib-reading service.

This package replaces ``image_processor.video_inference``. It has no HTTP
dependency: the pipeline owns the camera and posts finish events to the results
API through a durable sink, so no browser tab can ever drive race timing.
"""

__all__ = ["config", "capture", "detect", "ocr", "finish", "pipeline", "sink"]
