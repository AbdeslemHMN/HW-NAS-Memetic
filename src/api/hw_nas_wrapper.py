import os
import pickle
import re
from typing import Dict, Optional, Tuple

import numpy as np


class HWNASApi:
    """
    Interface for HW-NAS-Bench.
    Provides O(1) lookup for architecture hardware metrics on NAS-Bench-201.

    Accuracy lookup strategy (in priority order):
      1. Compact cache (.npz, ~1 MB) produced by scripts/extract_nas201_accuracy.py
         — ground-truth test accuracy from NAS-Bench-201, zero RAM pressure.
      2. Structural proxy — operation-richness estimate, always available.

    The 4.7 GB NAS-Bench-201-v1_1-*.pth file is never loaded at runtime.
    Run scripts/extract_nas201_accuracy.py once to produce the cache.
    """

    OPS = ["none", "skip_connect", "avg_pool_3x3", "nor_conv_1x1", "nor_conv_3x3"]

    def __init__(self, dataset_path: str, nas201_cache_path: Optional[str] = None):
        """
        :param dataset_path:      Path to HW-NAS-Bench-v1_0.pickle
        :param nas201_cache_path: Optional path to nas201_accuracy_cache.npz
                                  (produced by scripts/extract_nas201_accuracy.py).
                                  When omitted, the file is auto-detected from the
                                  same directory as dataset_path.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Benchmark file not found at {dataset_path}")

        print(f"Loading benchmark from {dataset_path}...")
        with open(dataset_path, "rb") as f:
            self.data = pickle.load(f)

        self.devices = list(self.data.keys())
        self._nasbench201_arch_to_index: Dict[Tuple[int, ...], int] = {}
        if "nasbench201" in self.data:
            self._build_nasbench201_index()

        # ── Accuracy cache (.npz, ~1 MB) ──────────────────────────────────────
        self._acc_cache: Optional[Dict[str, np.ndarray]] = None

        # Auto-detect cache next to the pickle if not explicitly provided
        if nas201_cache_path is None:
            _auto = os.path.join(os.path.dirname(dataset_path), "nas201_accuracy_cache.npz")
            if os.path.exists(_auto):
                nas201_cache_path = _auto

        if nas201_cache_path is not None:
            if not os.path.exists(nas201_cache_path):
                print(f"WARNING: accuracy cache not found at {nas201_cache_path} — using structural proxy.")
            else:
                _npz = np.load(nas201_cache_path)
                self._acc_cache = {k: _npz[k] for k in _npz.files}
                _kb = sum(v.nbytes for v in self._acc_cache.values()) // 1024
                print(f"Accuracy cache loaded ({_kb} KB) — ground-truth mode active.")

        if self._acc_cache is None:
            print("Accuracy mode: structural proxy  "
                  "(run scripts/extract_nas201_accuracy.py for ground-truth values)")

        print(f"Benchmark loaded. Available devices: {self.devices}")

    def _build_nasbench201_index(self) -> None:
        config_list = self.data["nasbench201"]["cifar10"].get("config", [])
        for idx, config in enumerate(config_list):
            arch_tuple = self._arch_str_to_tuple(config["arch_str"])
            self._nasbench201_arch_to_index[arch_tuple] = idx

    def _arch_str_to_tuple(self, arch_str: str) -> Tuple[int, ...]:
        groups = [group.strip("|") for group in arch_str.split("+") if group.strip("|")]
        op_names = []
        for group in groups:
            op_names.extend(re.findall(r"([a-z_0-9]+)~\d+", group))

        if len(op_names) != 6:
            raise ValueError(
                f"Invalid architecture string '{arch_str}'. Expected 6 ops, got {len(op_names)}."
            )

        try:
            return tuple(self.OPS.index(op_name) for op_name in op_names)
        except ValueError as exc:
            raise ValueError(f"Unknown operation in architecture string: {exc}") from exc

    # Structural accuracy proxy: weights per operation index (none→conv3x3)
    _OP_WEIGHTS = [0.0, 0.15, 0.55, 0.75, 1.0]
    _ACC_RANGE = {
        "cifar10":        (10.0, 94.0),
        "cifar100":       ( 5.0, 73.0),
        "ImageNet16-120": ( 3.0, 47.0),
    }

    def query_accuracy(self, architecture_vector, dataset: str = "cifar10") -> float:
        """
        Returns the test accuracy for the given architecture.

        Ground-truth mode (cache present):
          O(1) array lookup into the pre-extracted .npz cache.
          Cache holds the hp='200' final-epoch test accuracy for every
          NAS-Bench-201 architecture (15,625 entries × 3 datasets, ~1 MB total).

        Proxy mode (no cache):
          Structural estimate from operation richness, scaled to known accuracy
          ranges. Always available as a fallback.

        :param architecture_vector: 6-integer op-index vector
        :param dataset: 'cifar10' | 'cifar100' | 'ImageNet16-120'
        :return: test accuracy in percent (e.g. 93.56)
        """
        if len(architecture_vector) != 6:
            raise ValueError("Architecture vector must contain exactly 6 edges.")

        if self._acc_cache is not None:
            arch_tuple = tuple(int(x) for x in architecture_vector)
            idx = self._nasbench201_arch_to_index.get(arch_tuple)
            if idx is not None and dataset in self._acc_cache:
                return float(self._acc_cache[dataset][idx])

        # ── Structural proxy fallback ─────────────────────────────────────────
        lo, hi = self._ACC_RANGE.get(dataset, (10.0, 94.0))
        raw = sum(self._OP_WEIGHTS[int(op)] for op in architecture_vector) / 6.0
        return lo + raw * (hi - lo)

    def query(self, architecture_vector, device: str, dataset: str = "cifar10", metric: str = "edgegpu_latency") -> float:
        """
        Queries the benchmark for a specific NAS-Bench-201 architecture.

        :param architecture_vector: List or np.array of 6 integers (0-4)
        :param device: The top-level benchmark key, e.g. 'nasbench201'
        :param dataset: The dataset name inside the benchmark, e.g. 'cifar10'
        :param metric: The hardware metric, e.g. 'edgegpu_latency'
        :return: latency value for the requested architecture
        """
        if device not in self.data:
            raise ValueError(f"Device '{device}' not found. Choose from: {self.devices}")

        if len(architecture_vector) != 6:
            raise ValueError("Architecture vector must contain exactly 6 edges.")

        arch_tuple = tuple(int(x) for x in architecture_vector)
        if device == "nasbench201":
            if arch_tuple not in self._nasbench201_arch_to_index:
                raise KeyError(f"Architecture {arch_tuple} not found in NAS-Bench-201 benchmark.")

            idx = self._nasbench201_arch_to_index[arch_tuple]
            try:
                metric_array = self.data[device][dataset][metric]
            except KeyError as exc:
                raise KeyError(f"Dataset '{dataset}' or metric '{metric}' not found for '{device}'.") from exc

            return float(metric_array[idx])

        raise NotImplementedError(f"Query is only implemented for 'nasbench201', not '{device}'.")


if __name__ == "__main__":
    try:
        api = HWNASApi("data/HW-NAS-Bench-v1_0.pickle")
        sample_arch = [2, 3, 1, 3, 1, 1]
        latency  = api.query(sample_arch, device="nasbench201")
        accuracy = api.query_accuracy(sample_arch, dataset="cifar10")
        mode     = "ground-truth" if api._acc_cache is not None else "proxy"
        print(f"Arch: {sample_arch}")
        print(f"  Latency  : {latency:.4f} ms")
        print(f"  Accuracy : {accuracy:.2f}%  [{mode}]")
    except Exception as e:
        print(f"Setup test failed: {e}")