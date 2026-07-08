"""
app/utils/logger.py
───────────────────
Structured logging for the RAG system.

Every module gets its own named logger via get_logger(__name__).
All loggers share a single handler configured here, so log_level in
.env controls the entire application from one place.

Format:
    2024-01-15 14:32:01 | INFO     | app.core.retriever:42 | Retrieved 5 chunks

Usage:
    from app.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Retrieval complete", extra={"chunks": 5, "query": query[:60]})
    logger.warning("Low confidence", extra={"score": 0.61})
    logger.error("Gemini call failed", exc_info=True)
"""

import logging
import sys
from typing import Optional


# ── Formatter ─────────────────────────────────────────────────────────────────

class _PaddedLevelFormatter(logging.Formatter):
    """
    Custom formatter that:
    - Pads level names to a fixed width for visual alignment
    - Shows module path and line number
    - Uses ISO-style timestamps (no milliseconds for readability)
    """

    FMT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    DATEFMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self.FMT, datefmt=self.DATEFMT)


# ── Root handler (configured once) ────────────────────────────────────────────

def _build_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_PaddedLevelFormatter())
    return handler


_handler = _build_handler()
_handler_attached_to: set[str] = set()   # tracks which loggers already have the handler


# ── Public API ─────────────────────────────────────────────────────────────────

def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger for a module.

    Args:
        name:  Pass __name__ from the calling module. This gives hierarchical
               logger names like 'app.core.retriever', which makes log output
               immediately traceable to the source file.
        level: Override log level for this specific logger. If None, the level
               is read from settings (or defaults to INFO).

    Returns:
        A configured Logger instance. Calling this multiple times with the
        same name returns the same logger (Python's logging.getLogger is
        idempotent by name).
    """
    # Lazy import to avoid circular dependency with config.py at module load
    from app.config import settings

    logger = logging.getLogger(name)

    # Attach handler only once per logger name
    if name not in _handler_attached_to:
        logger.addHandler(_handler)
        logger.propagate = False        # prevent double-logging via root logger
        _handler_attached_to.add(name)

    effective_level = level or settings.log_level
    logger.setLevel(getattr(logging, effective_level, logging.INFO))

    return logger


def configure_uvicorn_logging() -> None:
    """
    Align uvicorn's access and error loggers with the application formatter
    so all log lines share the same format in the terminal.

    Call this once from app/main.py on startup.
    """
    from app.config import settings

    for uvicorn_logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uv_logger = logging.getLogger(uvicorn_logger_name)
        uv_logger.handlers.clear()
        uv_logger.addHandler(_handler)
        uv_logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
        uv_logger.propagate = False
