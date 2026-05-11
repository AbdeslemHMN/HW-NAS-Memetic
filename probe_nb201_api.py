r"""
probe_nb201_api.py
==================
Run this ONCE before anything else to find out exactly which
NAS-Bench-201 API version you have and what method names to use.

Usage:
    python probe_nb201_api.py <path_to_NAS-Bench-201.pth>

Example:
    python probe_nb201_api.py ".\NAS-Bench-201-v1_1-096897.pth"

If you don't have the file yet, see the download instructions printed below.
"""

import sys
import os

# ── Download instructions ────────────────────────────────────────────────────
DOWNLOAD_HELP = """
HOW TO GET NAS-Bench-201:
─────────────────────────
Option A — Direct download (easiest):
    https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_

    File: NAS-Bench-201-v1_1-096897.pth  (~2.5 GB)
    Put it in your project folder.

Option B — Via the package:
    pip install nas-bench-201
    python -c "from nas_201_api import NASBench201API; api = NASBench201API('NAS-Bench-201-v1_1-096897.pth')"

Option C — Clone the repo:
    git clone https://github.com/D-X-Y/NAS-Bench-201.git
    cd NAS-Bench-201
    pip install -e .
"""

if len(sys.argv) < 2:
    print(__doc__)
    print(DOWNLOAD_HELP)
    sys.exit(0)

pth_path = sys.argv[1]

# ── Check file exists ────────────────────────────────────────────────────────
print(f"\n[1] Checking file: {pth_path}")
if not os.path.exists(pth_path):
    print(f"    NOT FOUND.")
    print(DOWNLOAD_HELP)
    sys.exit(1)

size_mb = os.path.getsize(pth_path) / (1024 ** 2)
print(f"    Found. Size: {size_mb:.1f} MB")

# ── Try importing the API ────────────────────────────────────────────────────
print("\n[2] Importing NASBench201API ...")
try:
    from nas_201_api import NASBench201API
    print("    Import OK.")
except ImportError:
    print("    FAILED. Install with: pip install nas-bench-201")
    print("    Or: cd NAS-Bench-201 && pip install -e .")
    sys.exit(1)

# ── Load the API ─────────────────────────────────────────────────────────────
print("\n[3] Loading API (may take 10-30 seconds) ...")
try:
    import torch
    import numpy as np

    # 🔥 IMPORTANT: use _core (new numpy internal path)
    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar
    ])

    # 🔥 ALSO force unsafe load (required for this dataset)
    torch_load_original = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return torch_load_original(*args, **kwargs)

    torch.load = patched_torch_load

    api = NASBench201API(pth_path, verbose=False)
    print(f"    Loaded. {len(api)} architectures.")

except Exception as e:
    print(f"    ERROR: {e}")
    sys.exit(1)

# ── Probe available methods ──────────────────────────────────────────────────
print("\n[4] Available methods on the API object:")
methods = [m for m in dir(api) if not m.startswith('__')]
for m in methods:
    print(f"    {m}")

# ── Try a test architecture ──────────────────────────────────────────────────
TEST_ARCH_STR = '|nor_conv_3x3~0|+|none~0|skip_connect~1|+|nor_conv_1x1~0|nor_conv_1x1~1|nor_conv_1x1~2|'

print(f"\n[5] Probing with test arch:\n    {TEST_ARCH_STR}")

# --- Method A: query_index_by_arch + get_more_info ---
print("\n  [A] Trying: api.query_index_by_arch() + api.get_more_info()")
try:
    idx = api.query_index_by_arch(TEST_ARCH_STR)
    print(f"      arch index = {idx}")

    for dataset in ['cifar10', 'cifar10-valid', 'cifar100', 'ImageNet16-120']:
        try:
            info = api.get_more_info(idx, dataset=dataset, hp='200', is_random=False)
            print(f"      dataset={dataset!r:20s}  keys={list(info.keys())}")
            for k, v in info.items():
                if 'acc' in k.lower() or 'accuracy' in k.lower():
                    print(f"        → {k}: {v}")
        except Exception as e2:
            print(f"      dataset={dataset!r:20s}  ERROR: {e2}")
except Exception as e:
    print(f"      ERROR: {e}")

# --- Method B: query_by_arch ---
print("\n  [B] Trying: api.query_by_arch()")
try:
    result = api.query_by_arch(TEST_ARCH_STR, '200')
    print(f"      Result type: {type(result)}")
    print(f"      Result: {result}")
except Exception as e:
    print(f"      ERROR: {e}")

# --- Method C: api[idx] ---
print("\n  [C] Trying: api[0] (direct index access)")
try:
    item = api[0]
    print(f"      Type: {type(item)}")
    print(f"      Value: {repr(item)[:200]}")
except Exception as e:
    print(f"      ERROR: {e}")

# --- Method D: arch_str_list ---
print("\n  [D] Trying: api.arch_str_list")
try:
    arch_list = api.arch_str_list
    print(f"      Length: {len(arch_list)}")
    print(f"      First entry: {arch_list[0]}")
    print(f"      Index of test arch: {arch_list.index(TEST_ARCH_STR) if TEST_ARCH_STR in arch_list else 'not found'}")
except Exception as e:
    print(f"      ERROR: {e}")

print("\n[6] Done. Copy the output above and share it — "
      "it tells us exactly which method to use in the loader.")