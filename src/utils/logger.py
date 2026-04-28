"""
Logging and persistence utilities for HW-NAS-Memetic experiments.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a single StreamHandler (idempotent — safe to
    call multiple times with the same name).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def _arch_to_list(arch: Any) -> list[int]:
    """Convert ndarray | list | tuple arch to a plain Python list of ints."""
    if isinstance(arch, np.ndarray):
        return arch.tolist()
    return [int(x) for x in arch]


def save_archive(
    archive: list[dict],
    path: str | Path,
    metadata: dict | None = None,
) -> Path:
    """
    Persist an experiment archive to disk.

    File format is inferred from the extension:
      - .json  → nested JSON with metadata header
      - .csv   → flat CSV via pandas (metadata written as a sidecar .meta.json)

    :param archive:  List of {'arch', 'accuracy', 'latency'} dicts.
    :param path:     Output file path. Parent directories are created if absent.
    :param metadata: Optional dict merged into the JSON header / sidecar.
    :return:         Resolved absolute Path of the written file.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(tz=timezone.utc).isoformat()
    base_meta: dict = {"timestamp_utc": timestamp, "n_evaluations": len(archive)}
    if metadata:
        base_meta.update(metadata)

    # Serialise archive rows to pure-Python types
    rows = [
        {
            "arch": _arch_to_list(entry["arch"]),
            "accuracy": float(entry["accuracy"]),
            "latency": float(entry["latency"]),
        }
        for entry in archive
    ]

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.DataFrame(rows)
        df.insert(0, "index", range(len(df)))
        df.to_csv(path, index=False)
        # Write sidecar metadata
        meta_path = path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(base_meta, indent=2))
        logging.getLogger(__name__).debug("Sidecar metadata written to %s", meta_path)
    else:
        # Default: JSON
        payload = {"metadata": base_meta, "archive": rows}
        path.write_text(json.dumps(payload, indent=2))

    logging.getLogger(__name__).info(
        "Archive saved → %s  (%d entries)", path, len(archive)
    )
    return path
