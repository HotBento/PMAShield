"""
Centralised logging configuration using loguru.

Usage in any module
-------------------
    from pma_shield.logger import logger

Call ``setup_file_logging()`` once from an entry-point script to add a
persistent file sink under the ``log/`` directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

# ── Remove loguru's default stderr handler ───────────────────────────────────
logger.remove()

# ── Console sink: INFO and above, compact and coloured ───────────────────────
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    colorize=True,
)


def setup_file_logging(log_dir: str | Path = "log", run_name: str = "") -> str:
    """Add a DEBUG-level file sink with full source location info.

    Parameters
    ----------
    log_dir:  directory for log files (created if missing), default ``log/``
    run_name: optional label appended to the timestamp in the filename

    Returns
    -------
    str: absolute path of the created log file
    """
    from datetime import datetime

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{run_name}" if run_name else ""
    log_path = log_dir / f"{ts}{suffix}.log"

    logger.add(
        str(log_path),
        level="DEBUG",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | "
            "{name}:{function}:{line} | {message}"
        ),
        encoding="utf-8",
        enqueue=True,   # thread-safe for concurrent evaluations
    )

    logger.info("File logging → {}", log_path)
    return str(log_path)
