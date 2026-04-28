"""
Random Search baseline for HW-NAS-Memetic.

Research Insight:
- Uniform random sampling over {0..4}^6 (15,625 points) is an unbiased baseline
  and the theoretical lower bound on search quality for a given budget.
  Its expected hypervolume (HV) grows as O(B^{2/3}) for two objectives
  (Beume et al., 2007), providing a natural reference for measuring MOEA/D gain.
"""
from src.algorithms.base_optimizer import BaseOptimizer


class RandomSearch(BaseOptimizer):
    """Exhausts the full evaluation budget with uniformly sampled architectures."""

    def search(self) -> list[dict]:
        while self.budget_spent < self.budget:
            arch = self._random_arch()
            self._eval(arch)
        return self.global_archive
