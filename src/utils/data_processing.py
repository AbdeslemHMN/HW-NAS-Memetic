"""Shared filesystem and data-frame helpers for HW-NAS-Memetic.

This module contains the standard root discovery and JSON-backed result
loading utilities used by notebooks and downstream analysis code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def get_project_root() -> Path:
    """Return the repository root directory for the current working directory.

    Walks upward from ``Path.cwd()`` until a directory containing both ``src/``
    and ``requirements.txt`` is found.
    """
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "src").exists() and (candidate / "requirements.txt").exists():
            return candidate
    raise FileNotFoundError("Could not discover PROJECT_ROOT from current working directory.")


def load_json_file(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found.")
    return json.loads(path.read_text())


def load_archives_glob(results_dir: Path, glob_pattern: str) -> list[list[dict]]:
    archives: list[list[dict]] = []
    for path in sorted(Path(results_dir).glob(glob_pattern)):
        payload = load_json_file(path)
        archives.append(payload["archive"])
    return archives


def load_runs_glob(results_dir: Path, glob_pattern: str) -> list[dict]:
    return [load_json_file(path) for path in sorted(Path(results_dir).glob(glob_pattern))]


def load_archive(results_dir: Path, fname: str) -> list[dict]:
    path = Path(results_dir) / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/run_incremental_build.py --dataset <dataset> --hardware <hardware> --seed 42"
        )
    payload = load_json_file(path)
    return payload["archive"]


def load_incremental_archive(results_base: Path, dataset: str, hardware: str, fname: str) -> list[dict]:
    path = Path(results_base) / dataset / hardware / fname
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/run_incremental_build.py --dataset {dataset} --hardware {hardware} --seed 42"
        )
    payload = load_json_file(path)
    return payload["archive"]


def archive_to_dataframe(archive: list[dict]):
    import pandas as pd

    df = pd.DataFrame(archive)
    if "arch" in df.columns:
        df["arch"] = df["arch"].apply(lambda x: tuple(x))
    return df


def archives_to_dataframe(archives: list[list[dict]]):
    import pandas as pd

    return pd.concat([archive_to_dataframe(a) for a in archives], ignore_index=True) if archives else pd.DataFrame()
