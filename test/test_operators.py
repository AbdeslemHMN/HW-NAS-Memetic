import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.operators import GAOperator, MemeticState, PSOOperator, SAOperator


class TestOperators(unittest.TestCase):
    def test_ga_operator_returns_valid_architecture(self):
        rng = np.random.default_rng(0)
        state = MemeticState(
            current=rng.integers(0, 5, (4, 6)),
            pbest=rng.integers(0, 5, (4, 6)),
            gbest=rng.integers(0, 5, (4, 6)),
            accs=rng.random(4),
            lats=rng.random(4),
            scores=rng.random(4),
            weights=np.vstack([np.linspace(1, 0, 4), np.linspace(0, 1, 4)]).T,
            neighborhoods=np.column_stack([np.zeros(4, dtype=int), np.ones(4, dtype=int)]),
            T_sa=1.0,
        )

        candidate = GAOperator().apply(state, 0, rng)
        self.assertEqual(candidate.shape, (6,))
        self.assertIn(candidate.dtype.kind, {"i", "u"})
        self.assertTrue(np.all((0 <= candidate) & (candidate < 5)))

    def test_pso_operator_returns_valid_architecture(self):
        rng = np.random.default_rng(1)
        state = MemeticState(
            current=rng.integers(0, 5, (4, 6)),
            pbest=rng.integers(0, 5, (4, 6)),
            gbest=rng.integers(0, 5, (4, 6)),
            accs=rng.random(4),
            lats=rng.random(4),
            scores=rng.random(4),
            weights=np.vstack([np.linspace(1, 0, 4), np.linspace(0, 1, 4)]).T,
            neighborhoods=np.column_stack([np.zeros(4, dtype=int), np.ones(4, dtype=int)]),
            T_sa=1.0,
        )

        candidate = PSOOperator().apply(state, 2, rng)
        self.assertEqual(candidate.shape, (6,))
        self.assertIn(candidate.dtype.kind, {"i", "u"})
        self.assertTrue(np.all((0 <= candidate) & (candidate < 5)))

    def test_sa_operator_acceptance_and_cooling(self):
        rng = np.random.default_rng(2)
        sa = SAOperator(T0=1.0, alpha=0.95)

        self.assertTrue(sa.accept(0.5, 0.6, 1.0, rng))
        self.assertIsInstance(sa.accept(0.6, 0.5, 1.0, rng), bool)
        self.assertAlmostEqual(sa.cool(1.0), 0.95, places=9)


if __name__ == "__main__":
    unittest.main()
