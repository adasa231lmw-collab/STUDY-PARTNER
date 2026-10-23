"""Structured logging configuration."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from jarvis.config import settings


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("jarvis")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    try:
        if settings.LOG_FILE:
            fh = logging.FileHandler(str(settings.LOG_FILE), encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            logger.addHandler(fh)
    except Exception:
        pass
    return logger