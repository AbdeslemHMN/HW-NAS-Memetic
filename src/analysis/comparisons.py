"""
Genotype-level comparison utilities for HW-NAS-Memetic.

Provides set-algebra analysis on architecture strings (genotypes) found on the
Pareto fronts of different algorithms.  The core idea is to answer:

  "Does our Proposed algorithm discover architectures that NSGA-II misses?"

Convention
----------
Each archive entry is a ``dict`` with at minimum the keys:

  ``"accuracy"``   – float, validation accuracy on the target dataset
  ``"latency"``    – float, measured latency (ms) on the target hardware
  ``"arch_str"``   – str,  unique architecture genotype string
                     (also accepted as ``"arch_string"`` or ``"genotype"``)

If ``arch_str`` is absent the entry is silently skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from src.analysis.pareto_metrics import archive_to_points


# ── Efficient non-domination for 2-D objective space ──────────────────────────

def _nd_indices_2d(pts: np.ndarray) -> np.ndarray:
    """Return integer indices of non-dominated rows in a 2-D minimisation problem.

    O(N log N) via a lexicographic sweep — identical logic to the private
    ``_pareto_front_2d_sweep`` in ``pareto_metrics``, but preserving original
    indices so we can map back to archive metadata.

    :param pts: (N, 2) float array of [f1, f2] objective values.
    :return:    1-D integer array of non-dominated row indices (unsorted).
    """
    n = len(pts)
    if n == 0:
        return np.array([], dtype=np.intp)

    # Sort by (f1 asc, f2 asc)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    f2s   = pts[order, 1]

    # Sweep: track running minimum of f2; a point is non-dominated iff
    # its f2 <= running min so far (lower f2 = better).
    nd_flags = np.zeros(n, dtype=bool)
    best_f2  = np.inf
    for rank, orig_idx in enumerate(order):
        f2 = f2s[rank]
        if f2 <= best_f2:
            nd_flags[orig_idx] = True
            best_f2 = f2

    return np.where(nd_flags)[0]


# ── Helpers ───────────────────────────────────────────────────────────────────

_ARCH_KEY_CANDIDATES = ("arch_str", "arch_string", "genotype", "arch")


def _normalise_arch(s: str) -> str:
    """Canonical form: lower-case, no whitespace."""
    return s.lower().replace(" ", "").replace("\t", "")


def _extract_raw_arch(entry: dict) -> str | None:
    for key in _ARCH_KEY_CANDIDATES:
        if key in entry:
            return str(entry[key])
    return None


def _extract_arch(entry: dict) -> str | None:
    raw = _extract_raw_arch(entry)
    return _normalise_arch(raw) if raw is not None else None


def pareto_front_with_metadata(
    archive: list[dict],
) -> list[dict]:
    """Return only the non-dominated entries from *archive*, preserving all metadata.

    Uses the same 2-objective space as the rest of the project:
    ``F = [-accuracy, latency]``.

    :param archive: Raw per-seed or pooled archive (list of eval dicts).
    :return:        Subset of archive entries that lie on the Pareto front.
    """
    if not archive:
        return []

    pts    = archive_to_points(archive)          # (N, 2)  [-acc, lat]
    nd_idx = _nd_indices_2d(pts)                 # O(N log N), returns indices
    return [archive[i] for i in nd_idx]


def aggregate_front_with_metadata(
    archives: list[list[dict]],
) -> list[dict]:
    """Pool all seed archives and return the global non-dominated front.

    :param archives: List of per-seed archives.
    :return:         List of globally non-dominated evaluation dicts.
    """
    if not archives:
        return []
    pooled = [entry for arch in archives for entry in arch]
    return pareto_front_with_metadata(pooled)


# ── Core comparison dataclass ─────────────────────────────────────────────────

@dataclass
class GenotypeOverlapResult:
    """Results of a genotype-level comparison between two Pareto fronts."""

    proposed_total: int
    baseline_total: int
    shared_count: int
    unique_to_proposed_count: int
    unique_to_baseline_count: int
    novelty_score: float            # % of proposed front NOT in baseline
    shared_archs: frozenset[str]
    unique_to_proposed: frozenset[str]
    unique_to_baseline: frozenset[str]

    # Convenience properties
    @property
    def proposed_coverage(self) -> float:
        """Fraction of proposed front that is completely novel (0-100%)."""
        return self.novelty_score

    @property
    def overlap_fraction(self) -> float:
        """Fraction of proposed front that overlaps with baseline (0-100%)."""
        if self.proposed_total == 0:
            return 0.0
        return 100.0 * self.shared_count / self.proposed_total

    @property
    def shared_archs_list(self) -> list[str]:
        return sorted(self.shared_archs)

    @property
    def unique_to_proposed_list(self) -> list[str]:
        return sorted(self.unique_to_proposed)

    @property
    def unique_to_baseline_list(self) -> list[str]:
        return sorted(self.unique_to_baseline)


def analyze_genotype_overlap(
    proposed_front: list[dict],
    baseline_front: list[dict],
) -> GenotypeOverlapResult:
    """Compare two Pareto fronts at the architecture-string level.

    :param proposed_front:  List of eval dicts for the Proposed algorithm.
    :param baseline_front:  List of eval dicts for the baseline algorithm.
    :return:                :class:`GenotypeOverlapResult` with full set-algebra.
    """
    def _arch_mapping(front: list[dict]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for e in front:
            raw = _extract_raw_arch(e)
            if raw is None:
                continue
            key = _normalise_arch(raw)
            if key not in mapping:
                mapping[key] = raw
        return mapping

    proposed_map = _arch_mapping(proposed_front)
    baseline_map = _arch_mapping(baseline_front)

    p_keys = frozenset(proposed_map)
    b_keys = frozenset(baseline_map)

    shared_keys           = p_keys & b_keys
    unique_proposed_keys  = p_keys - b_keys
    unique_baseline_keys  = b_keys - p_keys

    shared_archs = frozenset(sorted(proposed_map[k] for k in shared_keys))
    unique_to_proposed = frozenset(sorted(proposed_map[k] for k in unique_proposed_keys))
    unique_to_baseline = frozenset(sorted(baseline_map[k] for k in unique_baseline_keys))

    if len(p_keys) > 0:
        novelty = 100.0 * len(unique_to_proposed) / len(p_keys)
    else:
        novelty = 0.0

    return GenotypeOverlapResult(
        proposed_total=len(p_keys),
        baseline_total=len(b_keys),
        shared_count=len(shared_archs),
        unique_to_proposed_count=len(unique_to_proposed),
        unique_to_baseline_count=len(unique_to_baseline),
        novelty_score=novelty,
        shared_archs=shared_archs,
        unique_to_proposed=unique_to_proposed,
        unique_to_baseline=unique_to_baseline,
    )


# ── Multi-baseline helper ─────────────────────────────────────────────────────

@dataclass
class MultiBaselineOverlap:
    """Aggregated overlap results for one Proposed front vs N baselines."""
    dataset: str
    hardware: str
    per_baseline: dict[str, GenotypeOverlapResult] = field(default_factory=dict)
    proposed_total: int = 0

    @property
    def all_baseline_archs(self) -> frozenset[str]:
        """Union of all architecture strings found by any baseline."""
        union: set[str] = set()
        for r in self.per_baseline.values():
            union |= r.shared_archs | r.unique_to_baseline
        return frozenset(union)

    @property
    def fully_unique_to_proposed(self) -> frozenset[str]:
        """Architectures in Proposed that were missed by ALL baselines."""
        if not self.per_baseline:
            return frozenset()
        first_unique = next(iter(self.per_baseline.values())).unique_to_proposed
        result: frozenset[str] = first_unique
        for r in self.per_baseline.values():
            result = result & r.unique_to_proposed
        return result

    @property
    def fully_unique_to_proposed_list(self) -> list[str]:
        return sorted(self.fully_unique_to_proposed)

    @property
    def global_novelty_score(self) -> float:
        """% of Proposed front not discovered by any single baseline."""
        if self.proposed_total == 0:
            return 0.0
        return 100.0 * len(self.fully_unique_to_proposed) / self.proposed_total


def analyze_all_baselines(
    dataset: str,
    hardware: str,
    proposed_archives: list[list[dict]],
    baselines: dict[str, list[list[dict]]],
) -> MultiBaselineOverlap:
    """Run :func:`analyze_genotype_overlap` for every baseline.

    :param dataset:           Dataset name (for labelling only).
    :param hardware:          Hardware name (for labelling only).
    :param proposed_archives: List of per-seed archives for Proposed.
    :param baselines:         ``{baseline_name: [seed_archive, ...]}``
    :return:                  :class:`MultiBaselineOverlap` result.
    """
    proposed_front = aggregate_front_with_metadata(proposed_archives)

    result = MultiBaselineOverlap(
        dataset=dataset,
        hardware=hardware,
        proposed_total=len({_extract_arch(e) for e in proposed_front
                            if _extract_arch(e) is not None}),
    )

    for bname, barchives in baselines.items():
        baseline_front = aggregate_front_with_metadata(barchives)
        result.per_baseline[bname] = analyze_genotype_overlap(
            proposed_front, baseline_front
        )

    return result
