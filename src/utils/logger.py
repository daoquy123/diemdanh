"""Loguru-based logger with sane defaults for training scripts."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def get_logger(log_file: str | Path | None = None, level: str = "INFO"):
    """Return a configured loguru logger.

    The first call sets up the global sinks; subsequent calls just return the
    same logger. Pass ``log_file`` to additionally tee output to a file.
    """
    global _CONFIGURED
    if not _CONFIGURED:
        logger.remove()
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
                "<level>{level: <8}</level> | "
                "<cyan>{name}:{line}</cyan> - <level>{message}</level>"
            ),
            colorize=True,
        )
        _CONFIGURED = True
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, rotation="10 MB", retention=5)
    return logger
