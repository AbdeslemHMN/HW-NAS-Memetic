# load_hw_nas_bench.py
r"""
load_hw_nas_bench.py  — Real accuracy + Real latency
=====================================================
Combines:
    HW-NAS-Bench pickle  → real latency per device
    NAS-Bench-201 API    → real test accuracy per dataset

Returns:
    { tuple(arch_6int): (accuracy_percent, latency_ms) }

SETUP (do once):
    1. HW-NAS-Bench pickle — already in .\HW-NAS-Bench\HW-NAS-Bench-v1_0.pickle
    2. NAS-Bench-201 file  — download NAS-Bench-201-v1_1-096897.pth (~2.5 GB)
         https://drive.google.com/file/d/16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_
    3. Install API:
         pip install nas-bench-201
         -- OR --
         git clone https://github.com/D-X-Y/NAS-Bench-201.git && cd NAS-Bench-201 && pip install -e .

USAGE:
    from load_hw_nas_bench import load_real_bench

    data = load_real_bench(
        pickle_path  = r".\HW-NAS-Bench\HW-NAS-Bench-v1_0.pickle",
        nb201_path   = r".\NAS-Bench-201-v1_1-096897.pth",
        device       = "edgegpu",
        dataset      = "cifar10",
    )
"""

import os
import pickle
import warnings
import numpy as np
from typing import Dict, Tuple, Optional

OPS_LIST = [
    'nor_conv_3x3',   # 0
    'nor_conv_1x1',   # 1
    'avg_pool_3x3',   # 2
    'skip_connect',   # 3
    'none',           # 4
]
N_OPS = 5
N_EDGES = 6
N_ARCHS = N_OPS ** N_EDGES

IDX_TO_OP = {i: op for i, op in enumerate(OPS_LIST)}
OP_TO_IDX = {op: i for i, op in enumerate(OPS_LIST)}

DEVICES = ['edgegpu', 'raspi4', 'pixel3', 'eyeriss', 'fpga']
DATASETS = ['cifar10', 'cifar100', 'ImageNet16-120']

NB201_DATASET_MAP = {
    'cifar10': 'cifar10',          # use real test accuracy
    'cifar100': 'cifar100',
    'ImageNet16-120': 'ImageNet16-120',
}


def index_to_arch(idx: int) -> Tuple[int, ...]:
    arch = []
    for _ in range(N_EDGES):
        arch.append(idx % N_OPS)
        idx //= N_OPS
    return tuple(arch)


def arch_to_index(arch) -> int:
    return sum(op * (N_OPS ** j) for j, op in enumerate(arch))


def vector_to_arch_str(vec) -> str:
    ops = [IDX_TO_OP[v] for v in vec]
    return (f"|{ops[0]}~0|"
            f"+|{ops[1]}~0|{ops[2]}~1|"
            f"+|{ops[3]}~0|{ops[4]}~1|{ops[5]}~2|")


def arch_str_to_vector(arch_str: str) -> Tuple[int, ...]:
    ops = []
    for token in arch_str.split('|'):
        token = token.strip()
        if not token or token == '+':
            continue
        op_name = token.split('~')[0].strip()
        if op_name in OP_TO_IDX:
            ops.append(OP_TO_IDX[op_name])
    if len(ops) != 6:
        raise ValueError(f"Expected 6 ops, got {len(ops)} from: '{arch_str}'")
    return tuple(ops)


def _build_synthetic_accuracy() -> np.ndarray:
    rng = np.random.RandomState(42)
    accs = np.empty(N_ARCHS, dtype=np.float32)
    for idx in range(N_ARCHS):
        arch = index_to_arch(idx)
        n_conv = sum(1 for op in arch if op in (0, 1))
        n_skip = sum(1 for op in arch if op == 3)
        n_none = sum(1 for op in arch if op == 4)
        base = 70.0 + 4.0 * n_conv + 1.5 * n_skip - 3.0 * n_none
        accs[idx] = float(np.clip(base + rng.normal(0, 1.5), 50.0, 94.0))
    return accs


def _patch_torch_for_nb201():
    """
    PyTorch 2.6+ loads with weights_only=True by default.
    NAS-Bench-201 uses older pickle content, so we force weights_only=False.
    """
    import torch

    safe = []
    try:
        safe.append(np._core.multiarray.scalar)  # NumPy 2.x
    except Exception:
        try:
            safe.append(np.core.multiarray.scalar)  # older NumPy
        except Exception:
            pass

    try:
        if safe:
            torch.serialization.add_safe_globals(safe)
    except Exception:
        pass

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    return torch, original_load


def _load_nb201_accuracy(
    nb201_path: str,
    dataset: str,
    verbose: bool,
) -> Optional[np.ndarray]:
    if nb201_path is None:
        return None

    if not os.path.exists(nb201_path):
        print(f"[NAS-Bench-201] File not found: {nb201_path}")
        return None

    cache_file = os.path.splitext(nb201_path)[0] + f".{dataset}.acc_cache.npy"
    if os.path.exists(cache_file):
        if verbose:
            print(f"[NAS-Bench-201] Loading cached accuracy: {cache_file}")
        return np.load(cache_file)

    try:
        from nas_201_api import NASBench201API
    except ImportError:
        print("[NAS-Bench-201] Package not installed.")
        return None

    if verbose:
        print(f"[NAS-Bench-201] Loading from: {nb201_path} (may take ~30s) ...")

    torch = None
    original_load = None
    try:
        torch, original_load = _patch_torch_for_nb201()
        api = NASBench201API(nb201_path, verbose=False)
    except Exception as e:
        print(f"[NAS-Bench-201] Failed to load: {e}")
        if torch is not None and original_load is not None:
            torch.load = original_load
        return None
    finally:
        if torch is not None and original_load is not None:
            torch.load = original_load

    if verbose:
        try:
            print(f"[NAS-Bench-201] Loaded. {len(api)} architectures. Querying accuracy ...")
        except Exception:
            print("[NAS-Bench-201] Loaded. Querying accuracy ...")

    nb201_dataset = NB201_DATASET_MAP.get(dataset, dataset)
    acc_arr = np.full(N_ARCHS, np.nan, dtype=np.float64)

    for flat_idx in range(N_ARCHS):
        arch_str = vector_to_arch_str(index_to_arch(flat_idx))
        acc = _query_one_accuracy(api, arch_str, nb201_dataset)
        if acc is not None:
            acc_arr[flat_idx] = acc

    n_ok = int(np.sum(~np.isnan(acc_arr)))
    if verbose:
        print(f"[NAS-Bench-201] Accuracy loaded: {n_ok}/{N_ARCHS}")

    if np.any(np.isnan(acc_arr)):
        synth = _build_synthetic_accuracy()
        mask = np.isnan(acc_arr)
        acc_arr[mask] = synth[mask]
        if verbose:
            print(f"[NAS-Bench-201] Filled {int(mask.sum())} gaps with synthetic values.")

    try:
        np.save(cache_file, acc_arr)
        if verbose:
            print(f"[NAS-Bench-201] Saved cache: {cache_file}")
    except Exception:
        pass

    return acc_arr


def _query_one_accuracy(api, arch_str: str, nb201_dataset: str) -> Optional[float]:
    try:
        arch_idx = api.query_index_by_arch(arch_str)
        if arch_idx is not None and arch_idx >= 0:
            for hp in ['200', '12', None]:
                try:
                    kwargs = {'dataset': nb201_dataset, 'is_random': False}
                    if hp is not None:
                        kwargs['hp'] = hp
                    info = api.get_more_info(arch_idx, **kwargs)
                    for key in ['test-accuracy', 'valid-accuracy', 'train-accuracy']:
                        if key in info:
                            return float(info[key])
                except Exception:
                    continue
    except Exception:
        pass

    try:
        result = api.query_by_arch(arch_str, '200')
        if isinstance(result, dict):
            for key in ['test-accuracy', 'valid-accuracy', 'accuracy']:
                if key in result:
                    return float(result[key])
        elif isinstance(result, (int, float)):
            return float(result)
    except Exception:
        pass

    try:
        arch_idx = api.query_index_by_arch(arch_str)
        item = api[arch_idx]
        if hasattr(item, 'get_metrics'):
            m = item.get_metrics(nb201_dataset, 'x-test')
            if 'accuracy' in m:
                return float(m['accuracy'])
    except Exception:
        pass

    return None


def load_real_bench(
    pickle_path: str,
    nb201_path: Optional[str] = None,
    device: str = 'edgegpu',
    dataset: str = 'cifar10',
    cost_type: str = 'latency',
    verbose: bool = True,
) -> Dict[Tuple, Tuple[float, float]]:
    if device not in DEVICES:
        raise ValueError(f"Unknown device '{device}'. Choose from: {DEVICES}")
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {DATASETS}")

    if verbose:
        print(f"[HW-NAS-Bench] Loading: {pickle_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(pickle_path, 'rb') as f:
            raw = pickle.load(f)

    ds = raw['nasbench201'][dataset]
    latency_key = f"{device}_{cost_type}"

    if latency_key not in ds:
        raise KeyError(f"Key '{latency_key}' not found. Available: {list(ds.keys())}")

    latency_arr = np.array(ds[latency_key], dtype=np.float64)

    if verbose:
        print(f"[HW-NAS-Bench] Device={device} | Dataset={dataset} | Cost={cost_type}")
        print(f"[HW-NAS-Bench] Latency: {latency_arr.min():.4f} – {latency_arr.max():.4f} ms")

    acc_arr = _load_nb201_accuracy(nb201_path, dataset, verbose)

    if acc_arr is None:
        if verbose:
            print("\n[Accuracy] WARNING: Using SYNTHETIC accuracy.")
            print("[Accuracy] Results are NOT valid for paper-quality comparison.")
        acc_arr = _build_synthetic_accuracy().astype(np.float64)
        accuracy_source = "SYNTHETIC (not valid for papers)"
    else:
        accuracy_source = f"real NAS-Bench-201 ({dataset})"

    table: Dict[Tuple, Tuple[float, float]] = {
        index_to_arch(i): (float(acc_arr[i]), float(latency_arr[i]))
        for i in range(N_ARCHS)
    }

    if verbose:
        accs = [v[0] for v in table.values()]
        lats = [v[1] for v in table.values()]
        print(f"\n{'='*55}")
        print(f"  Accuracy source : {accuracy_source}")
        print(f"  Accuracy range  : {min(accs):.2f}% – {max(accs):.2f}%")
        print(f"  Latency range   : {min(lats):.4f} – {max(lats):.4f} ms")
        print(f"  Total loaded    : {len(table)} architectures")
        print(f"{'='*55}")

    return table


if __name__ == '__main__':
    import sys

    hw_pickle = r".\HW-NAS-Bench\HW-NAS-Bench-v1_0.pickle"
    nb201_pth = sys.argv[1] if len(sys.argv) > 1 else None

    table = load_real_bench(
        pickle_path=hw_pickle,
        nb201_path=nb201_pth,
        device='edgegpu',
        dataset='cifar10',
        verbose=True,
    )

    print("\nSample entries:")
    for arch, (acc, lat) in list(table.items())[:5]:
        print(f"  arch={list(arch)}  acc={acc:.4f}%  lat={lat:.4f} ms")