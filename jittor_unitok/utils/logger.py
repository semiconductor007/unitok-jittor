"""Simple file and console logger."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterable


def setup_logger(log_path: str | Path) -> logging.Logger:
    """Create a logger writing both stdout and a log file."""

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(log_path))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def append_csv(csv_path: str | Path, row: Dict[str, object], fieldnames: Iterable[str]) -> None:
    """Append one row to a CSV file, writing the header if needed."""

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

