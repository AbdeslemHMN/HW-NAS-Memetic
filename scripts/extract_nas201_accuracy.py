"""
scripts/extract_nas201_accuracy.py
──────────────────────────────────
Extracts NAS-Bench-201 accuracy into a compact cache (~1 MB) using
NATS-Bench TSS — the official successor that stores each architecture as
an individual file, enabling on-demand loading with ~50 MB RAM.

Why NATS-Bench TSS instead of NAS-Bench-201 .pth?
──────────────────────────────────────────────────
NAS-Bench-201-v1_1-096897.pth  : 4.7 GB on disk → 9–11 GB RAM during
                                  torch.load/unpickling (OOM on most machines)
NATS-tss-v1_0-3ffb9-simple/   : ~300 MB on disk → ~2 MB RAM constant
                                  (fast_mode=True + per-arch cache eviction)

The TSS (Topology Search Space) in NATS-Bench is identical to NAS-Bench-201:
  • same 15,625 cell architectures
  • same three datasets (CIFAR-10, CIFAR-100, ImageNet16-120)
  • same training protocol, additional random seeds

Setup (one-time)
────────────────
  Step 1 – download the archive:
    python scripts/download_data.py --nats-bench
    — OR —
    wget -O data/NATS-tss.tar \\
      "https://www.dropbox.com/sh/ceeo70u1buow681/AAC2M-SbKOxiIqpB0UCgXNxja/NATS-tss-v1_0-3ffb9-simple.tar?dl=1"
    tar xf data/NATS-tss.tar -C data/

  Step 2 – install package:
    pip install nats_bench

  Step 3 – run this script:
    python scripts/extract_nas201_accuracy.py

Result: data/nas201_accuracy_cache.npz (~1 MB).
HWNASApi auto-detects it at next startup; the 300 MB archive is no longer needed.
"""

import argparse
import gc
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).resolve().parents[1]
DATA_DIR          = PROJECT_ROOT / "data"
CACHE_OUT         = DATA_DIR / "nas201_accuracy_cache.npz"

_NATS_SIMPLE_NAME = "NATS-tss-v1_0-3ffb9-simple"

# Candidate locations where the extracted archive might live
_NATS_CANDIDATES: List[Path] = [
    DATA_DIR / _NATS_SIMPLE_NAME,
    Path.home() / ".torch" / _NATS_SIMPLE_NAME,
    Path("/tmp") / _NATS_SIMPLE_NAME,
]

DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
N_ARCHS  = 15_625
DEFAULT_HP = "200"   # full 200-epoch training


# ─────────────────────────────────────────────────────────────────────────────

def _find_nats_path(hint: Optional[str]) -> Optional[Path]:
    """Return the NATS-Bench simple directory, searching known locations."""
    if hint:
        p = Path(hint)
        return p if p.exists() else None
    for candidate in _NATS_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _check_prereqs(nats_path: Path) -> None:
    errors: List[str] = []
    if not nats_path.is_dir():
        errors.append(f"  • NATS-Bench simple dir not found: {nats_path}")
    try:
        from nats_bench import create  # type: ignore  # noqa: F401
    except ImportError:
        errors.append(
            "  • nats_bench not installed.\n"
            "    Run:  pip install nats_bench"
        )
    if errors:
        print("Pre-flight check failed:\n" + "\n".join(errors))
        sys.exit(1)


def _print_download_guide() -> None:
    print(f"""
╔══════════════════════════════════════════════════════════════════════════╗
║        Download NATS-Bench TSS simple archive (~300 MB)                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Option A — automatic helper script (recommended):                       ║
║    pip install gdown                                                     ║
║    python scripts/download_data.py --nats-bench                          ║
║                                                                          ║
║  Option B — gdown directly (Google Drive):                               ║
║    pip install gdown                                                     ║
║    gdown 17_saCsj_krKjlCBLOJEpNtzPXArMCqxU -O data/NATS-tss.tar        ║
║    tar xf data/NATS-tss.tar -C data/                                     ║
║                                                                          ║
║  Option C — browser download (Google Drive folder):                      ║
║    https://drive.google.com/drive/folders/                               ║
║    1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt                                    ║
║    Download: NATS-tss-v1_0-3ffb9-simple.tar  then  tar xf … -C data/    ║
║                                                                          ║
║  Option D — OneDrive:                                                    ║
║    https://1drv.ms/u/s!Aqkc27lrowWDf6SvuIkSXx0UQaI?e=nfvM5r            ║
║                                                                          ║
║  After extraction, data/{_NATS_SIMPLE_NAME}/  should exist.              ║
╚══════════════════════════════════════════════════════════════════════════╝
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract NAS-Bench-201 accuracy to a compact cache using "
            "NATS-Bench TSS (fast_mode=True, ~50 MB RAM, no torch required)."
        )
    )
    parser.add_argument(
        "--nats-path", type=str, default=None,
        metavar="DIR",
        help=(
            f"Path to the extracted '{_NATS_SIMPLE_NAME}' directory. "
            f"Auto-detected if placed in data/ or $TORCH_HOME."
        ),
    )
    parser.add_argument(
        "--hp", type=str, default=DEFAULT_HP,
        help="Training-epoch key (default: '200').  Try '12' for a quick smoke-test.",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Print download instructions and exit.",
    )
    args = parser.parse_args()

    # ── Download guide ────────────────────────────────────────────────────
    if args.download:
        _print_download_guide()
        return

    # ── Already cached? ───────────────────────────────────────────────────
    if CACHE_OUT.exists():
        size_kb = CACHE_OUT.stat().st_size // 1024
        print(f"Cache already exists: {CACHE_OUT}  ({size_kb} KB)")
        print("Delete it and re-run to regenerate.")
        return

    # ── Locate NATS-Bench simple directory ────────────────────────────────
    nats_path = _find_nats_path(args.nats_path)
    if nats_path is None:
        print(f"ERROR: '{_NATS_SIMPLE_NAME}' not found in any of:")
        for c in _NATS_CANDIDATES:
            print(f"  {c}")
        _print_download_guide()
        sys.exit(1)

    print(f"NATS-Bench TSS path : {nats_path}")
    _check_prereqs(nats_path)

    # ── Create API (fast_mode = one file per arch, ~50 MB RAM) ───────────
    from nats_bench import create  # type: ignore

    print("Loading NATS-Bench API in fast_mode=True …")
    print("  (each architecture is a separate file — no bulk load, ~50 MB RAM constant)\n")
    api = create(str(nats_path), "tss", fast_mode=True, verbose=False)

    # ── Extract accuracy ──────────────────────────────────────────────────
    # NOTE: arch2infos_dict accumulates ~1.76 MB per architecture in RAM.
    # With 15,625 arches × 3 datasets that would exhaust all system memory.
    # Fix: query all datasets for one arch, then evict it from the cache
    # immediately.  RAM stays constant at ~2 MB instead of growing to ~27 GB.
    hp = args.hp
    print(f"hp='{hp}'  |  {N_ARCHS} architectures × {len(DATASETS)} datasets\n")

    merged: Dict[str, np.ndarray] = {
        ds: np.zeros(N_ARCHS, dtype=np.float32) for ds in DATASETS
    }
    failed_per_ds: Dict[str, int] = {ds: 0 for ds in DATASETS}

    print("  [all datasets — arch-major order, cache evicted per arch]\n", flush=True)
    for idx in range(N_ARCHS):
        if idx % 3000 == 0:
            print(f"  {idx:5d} / {N_ARCHS} …", flush=True)

        for ds in DATASETS:
            try:
                info = api.get_more_info(idx, ds, hp=hp, is_random=False)
                merged[ds][idx] = float(info.get("test-accuracy") or 0.0)
            except Exception:
                failed_per_ds[ds] += 1
                merged[ds][idx] = 0.0

        # Evict this arch from the internal cache to keep RAM constant.
        if idx in api.arch2infos_dict:
            del api.arch2infos_dict[idx]

        # Periodic GC to return freed pages to the OS.
        if idx % 500 == 0:
            gc.collect()

    print()
    for ds in DATASETS:
        accs  = merged[ds]
        valid = accs[accs > 0]
        if len(valid):
            print(
                f"  [{ds}] min={valid.min():.2f}%  max={valid.max():.2f}%  "
                f"mean={valid.mean():.2f}%  failed={failed_per_ds[ds]}"
            )
        else:
            print(
                f"  [{ds}] WARNING: all values are 0. "
                f"Try --hp 12 or verify the archive is complete."
            )
    print()

    # ── Save cache ────────────────────────────────────────────────────────
    np.savez_compressed(str(CACHE_OUT), **merged)
    size_kb = CACHE_OUT.stat().st_size / 1024
    print(f"Cache saved → {CACHE_OUT}  ({size_kb:.1f} KB)")
    print("HWNASApi auto-detects this file at next startup.")
    print("The NATS-Bench archive can now be deleted to reclaim ~300 MB.")


if __name__ == "__main__":
    main()
