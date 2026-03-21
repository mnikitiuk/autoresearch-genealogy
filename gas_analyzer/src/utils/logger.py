"""Structured logging setup using loguru."""

from __future__ import annotations

import sys
from pathlib import Path


def setup_logger(level: str = "INFO", log_file: str | None = None) -> None:
    """Configure loguru with console and optional file output."""
    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level=level, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                log_file,
                level=level,
                rotation="10 MB",
                retention="30 days",
                compression="zip",
            )
    except ImportError:
        import logging
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        )
