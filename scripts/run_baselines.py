#!/usr/bin/env python3
"""
Baseline sweep: RandomSearch and NSGA-II over 5 independent seeds.

Output: results/baselines/{algorithm}_seed{seed}.json
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.api.hw_nas_wrapper import HWNASApi
from src.algorithms.random_search import RandomSearch
from src.algorithms.nsga2_search import nsga2_search
from src.utils.logger import get_logger, save_archive

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH   = PROJECT_ROOT / "data" / "HW-NAS-Bench-v1_0.pickle"
RESULTS_DIR = PROJECT_ROOT / "results" / "baselines"
SEEDS = [0, 1, 2, 3, 4]
BUDGET = 200        # evaluations per run  (raise for full experiments)
POP_SIZE = 20       # NSGA-II population size
METRIC = "edgegpu_latency"   # hardware metric key within nasbench201
DATASET = "cifar10"

log = get_logger("baselines")


def build_eval_fn(api: HWNASApi, metric: str, dataset: str):
    """Wrap HWNASApi.query into the (accuracy, latency) contract."""
    def _eval(arch: np.ndarray) -> tuple[float, float]:
        latency  = api.query(arch, device="nasbench201", dataset=dataset, metric=metric)
        accuracy = api.query_accuracy(arch, dataset=dataset)
        return accuracy, latency

    return _eval


def main() -> None:
    log.info("Loading HW-NAS-Bench from %s", DATA_PATH)
    api = HWNASApi(str(DATA_PATH))  # auto-detects nas201_accuracy_cache.npz if present
    eval_fn = build_eval_fn(api, METRIC, DATASET)

    # ── RandomSearch ─────────────────────────────────────────────────────────
    for seed in SEEDS:
        log.info("[RandomSearch] seed=%d  budget=%d", seed, BUDGET)
        rs = RandomSearch(eval_fn, budget=BUDGET, rng=np.random.default_rng(seed))
        archive = rs.search()
        out_path = RESULTS_DIR / f"random_search_seed{seed}.json"
        save_archive(
            archive,
            out_path,
            metadata={"algorithm": "RandomSearch", "seed": seed,
                       "budget": BUDGET, "metric": METRIC, "dataset": DATASET},
        )
        log.info("[RandomSearch] seed=%d → %s", seed, out_path)

    # ── NSGA-II ───────────────────────────────────────────────────────────────
    for seed in SEEDS:
        log.info("[NSGA2] seed=%d  budget=%d  pop_size=%d", seed, BUDGET, POP_SIZE)
        archive = nsga2_search(eval_fn, budget=BUDGET,
                               pop_size=POP_SIZE, rng_seed=seed)
        out_path = RESULTS_DIR / f"nsga2_seed{seed}.json"
        save_archive(
            archive,
            out_path,
            metadata={"algorithm": "NSGA2", "seed": seed,
                       "budget": BUDGET, "metric": METRIC, "dataset": DATASET,
                       "pop_size": POP_SIZE},
        )
        log.info("[NSGA2] seed=%d → %s", seed, out_path)

    log.info("Baseline sweep complete. Results in %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
