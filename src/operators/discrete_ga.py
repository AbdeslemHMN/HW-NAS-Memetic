"""
Discrete Genetic Algorithm operators for NAS-Bench-201 search space.
Architecture vector: L=6 edges, values in {0, 1, 2, 3, 4}.

Research Insights:
- mutate: Per-gene Bernoulli trials with p_m=1/L preserve expected Hamming
  distance of 1 per generation, preventing hyper-disruption while ensuring
  ergodicity across the full search space.
- crossover: Single-point at k in [1, 5] avoids trivially copying one parent,
  which would collapse genetic diversity. Uniform crossover was rejected due
  to higher epistasis destruction risk on NAS cell DAGs.
"""
import numpy as np

OPS_COUNT = 5  # {0, 1, 2, 3, 4}
ARCH_LEN = 6


def mutate(arch: np.ndarray, p_m: float = 1.0 / ARCH_LEN, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Per-gene mutation: each edge is independently mutated with probability p_m.
    A mutated edge is assigned a uniformly random op != current op.

    :param arch:  Shape (6,) integer array, values in [0, 4].
    :param p_m:   Per-gene mutation probability. Default = 1/L.
    :param rng:   Optional seeded Generator for reproducibility.
    :return:      Mutated architecture, shape (6,).
    """
    assert arch.shape == (ARCH_LEN,), f"Expected shape ({ARCH_LEN},), got {arch.shape}"
    assert arch.min() >= 0 and arch.max() < OPS_COUNT, "Op values must be in [0, 4]."

    rng = rng or np.random.default_rng()
    child = arch.copy()
    mask = rng.random(ARCH_LEN) < p_m  # shape (6,) bool

    for j in np.where(mask)[0]:
        candidates = np.delete(np.arange(OPS_COUNT), child[j])
        child[j] = rng.choice(candidates)

    return child


def crossover(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-point crossover at a random cut-point k in [1, 5].

    :param p1:  Parent 1, shape (6,), values in [0, 4].
    :param p2:  Parent 2, shape (6,), values in [0, 4].
    :param rng: Optional seeded Generator for reproducibility.
    :return:    (child1, child2) tuple, each shape (6,).
    """
    assert p1.shape == p2.shape == (ARCH_LEN,), "Both parents must have shape (6,)."

    rng = rng or np.random.default_rng()
    k = rng.integers(1, ARCH_LEN)  # k in [1, 5]

    c1 = np.concatenate([p1[:k], p2[k:]])
    c2 = np.concatenate([p2[:k], p1[k:]])

    return c1, c2
