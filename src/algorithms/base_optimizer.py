"""
Abstract base class for all HW-NAS-Memetic search algorithms.

Architecture vector: L=6 edges, values in {0, 1, 2, 3, 4}.
eval_fn contract: Callable[[np.ndarray], tuple[float, float]]
  - Input : architecture vector, shape (6,), dtype int64
  - Output: (accuracy: float, latency: float)

Research Insights:
- global_archive uses list.append() — O(1) amortised, memory-efficient
  (avoids repeated NumPy concatenation at O(N) cost).
- budget_spent is a scalar counter; no locking needed in single-threaded runs.
"""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

ARCH_LEN = 6
OPS_COUNT = 5  # {0, 1, 2, 3, 4}


class BaseOptimizer(ABC):
    def __init__(
        self,
        eval_fn: Callable[[np.ndarray], tuple[float, float]],
        budget: int,
        rng: np.random.Generator | None = None,
    ):
        if budget < 1:
            raise ValueError(f"Budget must be >= 1, got {budget}.")
        self.eval_fn = eval_fn
        self.budget: int = budget
        self.budget_spent: int = 0
        self.global_archive: list[dict] = []
        self.rng: np.random.Generator = rng or np.random.default_rng()
        # NFE checkpoints: list of NFE values at which a snapshot was taken.
        # Recorded every _snapshot_interval evaluations. The notebook uses these
        # to reconstruct the HV trajectory by slicing archive[:nfe].
        self.nfe_checkpoints: list[int] = []
        self._snapshot_interval: int = 100

    def _eval(self, arch: np.ndarray) -> tuple[float, float]:
        """
        Evaluate arch, increment NFE counter, log entry to global_archive.

        Raises RuntimeError immediately if the budget is already exhausted —
        this enforces a strict NFE cutoff: the moment budget_spent == budget
        the search loop MUST stop without issuing further evaluations.

        :return: (accuracy, latency) as floats.
        """
        if self.budget_spent >= self.budget:
            raise RuntimeError(
                f"Budget hard-stop: _eval called at NFE={self.budget_spent} "
                f"(budget={self.budget}). All search loops must guard with "
                "'if self.budget_spent >= self.budget: break' before calling _eval."
            )
        accuracy, latency = self.eval_fn(arch)
        self.budget_spent += 1
        self.global_archive.append(
            {
                "arch": arch.copy(),
                "accuracy": float(accuracy),
                "latency": float(latency),
            }
        )
        # Record NFE checkpoint every _snapshot_interval evaluations.
        # The HV trajectory is reconstructed in post-processing by computing
        # HV on archive[:nfe] for each nfe in nfe_checkpoints.
        if self.budget_spent % self._snapshot_interval == 0:
            self.nfe_checkpoints.append(self.budget_spent)
        return float(accuracy), float(latency)

    def _random_arch(self) -> np.ndarray:
        """Sample a uniform random architecture from {0..4}^6, dtype int64."""
        return self.rng.integers(0, OPS_COUNT, size=ARCH_LEN, dtype=np.int64)

    @abstractmethod
    def search(self) -> list[dict]:
        """
        Run the search until budget is exhausted.
        :return: global_archive — list of {'arch', 'accuracy', 'latency'} dicts.
        """
