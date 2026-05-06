import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.memetic_nas import MemeticNAS
from src.algorithms.operators import GAOperator, PSOOperator, SAOperator


class TestMemeticNAS(unittest.TestCase):
    def test_memetic_nas_obeys_budget_and_returns_archive(self):
        def eval_fn(arch: np.ndarray) -> tuple[float, float]:
            total = int(arch.sum())
            return 100.0 - float(total), float(total)

        optimizer = MemeticNAS(
            eval_fn=eval_fn,
            budget=20,
            candidate_ops=[GAOperator(), PSOOperator()],
            sa_op=SAOperator(T0=1.0, alpha=0.95),
            K=3,
            T_neighborhood=2,
            rng=np.random.default_rng(0),
        )

        archive = optimizer.search()
        self.assertEqual(optimizer.budget_spent, 20)
        self.assertEqual(len(archive), 20)
        self.assertTrue(all("arch" in entry and "accuracy" in entry and "latency" in entry for entry in archive))
        self.assertTrue(all(isinstance(entry["arch"], np.ndarray) for entry in archive))


if __name__ == "__main__":
    unittest.main()
