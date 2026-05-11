"""
test_ga_operator.py
===================
Standalone test and comparison harness for all GAOperator variants.

Runs entirely on synthetic data — no HW-NAS-Bench pickle, no NAS-Bench-201
file, no full pipeline needed.  Pure numpy.

What it covers
--------------
1. Unit tests     — correctness checks for every improvement flag.
2. Variant benchmark — 2000-evaluation simulated run per variant,
                       measuring HV proxy, Pareto size, clone rate,
                       mean Hamming distance, and convergence speed.
3. Comparison table — printed side-by-side, same format as the project
                      report table.
4. Convergence plot — HV vs evaluations for every variant (matplotlib,
                      optional — skipped gracefully if not installed).

RUN:
    python test_ga_operator.py            # full suite
    python test_ga_operator.py --quick    # 500 evals, no plot
    python test_ga_operator.py --plot     # save convergence PNG
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Minimal stubs so the file runs without the full project tree
# ---------------------------------------------------------------------------

class MemeticState:
    """Minimal stub matching base_operator.MemeticState interface."""
    def __init__(self, population: List[np.ndarray]) -> None:
        self.current = population   # list of arch arrays


# Stub for src.operators.discrete_ga.mutate
def _mutate_stub(arch: np.ndarray, p_m: float, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli per-gene mutation on {0..4}^6."""
    child = arch.copy()
    for i in range(len(child)):
        if rng.random() < p_m:
            ops = [o for o in range(5) if o != child[i]]
            child[i] = int(rng.choice(ops))
    return child


# ---------------------------------------------------------------------------
# Inline GAOperator (copy of ga_operator.py without project imports)
# This lets the test file run standalone.
# ---------------------------------------------------------------------------

_N_OPS   = 5
_N_EDGES = 6
_DEFAULT_P_M      = 1.0 / _N_EDGES
_DEFAULT_P_M_HIGH = 3.0 / _N_EDGES
_DEFAULT_P_M_LOW  = 0.5 / _N_EDGES
_MAX_RESAMPLE     = 20


class GAOperator:
    """Inline copy of ga_operator.GAOperator for standalone testing."""

    def __init__(
        self,
        p_m: float = _DEFAULT_P_M,
        *,
        force_change: bool = True,
        adaptive: bool = False,
        p_m_high: float = _DEFAULT_P_M_HIGH,
        p_m_low: float = _DEFAULT_P_M_LOW,
        crossover: bool = False,
        freq_bias: bool = False,
    ) -> None:
        self.p_m = p_m
        self.force_change = force_change
        self.adaptive = adaptive
        self.p_m_high = p_m_high
        self.p_m_low = p_m_low
        self.crossover = crossover
        self.freq_bias = freq_bias
        self._budget_fraction: float = 0.0
        self._op_counts = np.ones((_N_EDGES, _N_OPS), dtype=float)

    def notify_budget(self, spent: int, total: int) -> None:
        if total > 0:
            self._budget_fraction = min(1.0, spent / total)

    def update_archive_stats(self, archive: List[np.ndarray]) -> None:
        counts = np.ones((_N_EDGES, _N_OPS), dtype=float)
        for arch in archive:
            for edge, op in enumerate(arch):
                counts[edge, int(op)] += 1.0
        self._op_counts = counts

    def apply(self, state: MemeticState, k: int, rng: np.random.Generator) -> np.ndarray:
        parent = state.current[k].copy()
        base = self._crossover(parent, state, k, rng) if self.crossover else parent
        pm = self._active_pm()
        child = self._mutate(base, pm, rng)
        if self.force_change:
            attempts = 0
            while np.array_equal(child, parent) and attempts < _MAX_RESAMPLE:
                child = self._mutate(base, pm, rng)
                attempts += 1
            if np.array_equal(child, parent):
                child = parent.copy()
                edge = int(rng.integers(0, _N_EDGES))
                ops = [o for o in range(_N_OPS) if o != parent[edge]]
                child[edge] = int(rng.choice(ops))
        return child

    def _active_pm(self) -> float:
        if not self.adaptive:
            return self.p_m
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * self._budget_fraction))
        return self.p_m_low + (self.p_m_high - self.p_m_low) * cos_factor

    def _mutate(self, arch: np.ndarray, pm: float, rng: np.random.Generator) -> np.ndarray:
        if not self.freq_bias:
            return _mutate_stub(arch, p_m=pm, rng=rng)
        child = arch.copy()
        for edge in range(_N_EDGES):
            if rng.random() < pm:
                counts = self._op_counts[edge].copy()
                counts[int(child[edge])] = 0.0
                inv_w = 1.0 / (counts + 1e-9)
                inv_w[int(child[edge])] = 0.0
                probs = inv_w / inv_w.sum()
                child[edge] = int(rng.choice(_N_OPS, p=probs))
        return child

    def _crossover(self, parent, state, k, rng):
        n = len(state.current)
        if n < 2:
            return parent.copy()
        others = [i for i in range(n) if i != k]
        k2 = int(rng.choice(others))
        donor = state.current[k2]
        mask = rng.random(_N_EDGES) < 0.5
        return np.where(mask, parent, donor).astype(parent.dtype)

    @property
    def active_pm(self) -> float:
        return self._active_pm()

    def variant_name(self) -> str:
        flags = []
        if self.force_change:
            flags.append("force_change")
        if self.adaptive:
            flags.append("adaptive_pm")
        if self.crossover:
            flags.append("crossover")
        if self.freq_bias:
            flags.append("freq_bias")
        return "GAOperator[" + (", ".join(flags) if flags else "baseline_only") + "]"


# ---------------------------------------------------------------------------
# Synthetic benchmark  (proxy for real HW-NAS-Bench lookup)
# ---------------------------------------------------------------------------

def _synthetic_objectives(arch: np.ndarray) -> Tuple[float, float]:
    """
    Deterministic proxy objectives for a length-6 {0..4}^6 architecture.

    Returns (accuracy, latency) — values in realistic ranges so the HV
    calculation is meaningful.

    accuracy  : 80–95 %   (higher is better)
    latency   : 1–30 ms   (lower is better)

    Designed so there is a genuine Pareto trade-off:
    low-latency archs tend to have lower accuracy.
    """
    x = arch.astype(float)
    # Accuracy: penalise op-0 (skip), reward op-4 (most expensive)
    acc_raw = (
        10.0 * x[0]
        + 8.0 * x[1]
        + 6.0 * x[2]
        + 9.0 * x[3]
        + 7.0 * x[4]
        + 5.0 * x[5]
    )
    accuracy = 80.0 + 15.0 * acc_raw / (4 * 6 * 10)   # scale to [80, 95]

    # Latency: sum of ops gives a proxy FLOPs count
    lat_raw = float(x.sum())                              # 0..24
    latency = 1.0 + 29.0 * lat_raw / 24.0               # scale to [1, 30]
    return float(accuracy), float(latency)


def _is_dominated(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    """Return True if solution a is dominated by solution b."""
    # b dominates a if b is at least as good on all objectives and strictly
    # better on at least one.  (maximise acc, minimise lat)
    return (b[0] >= a[0] and b[1] <= a[1]) and (b[0] > a[0] or b[1] < a[1])


def _pareto_front(
    solutions: List[Tuple[np.ndarray, float, float]]
) -> List[Tuple[np.ndarray, float, float]]:
    """Extract non-dominated solutions from (arch, acc, lat) list."""
    front = []
    for sol in solutions:
        dominated = False
        for other in solutions:
            if other is sol:
                continue
            if _is_dominated((sol[1], sol[2]), (other[1], other[2])):
                dominated = True
                break
        if not dominated:
            front.append(sol)
    return front


def _normalize_objectives(
    front: List[Tuple[np.ndarray, float, float]],
    acc_min: float = 80.0,
    acc_max: float = 95.0,
    lat_min: float = 1.0,
    lat_max: float = 30.0,
) -> List[Tuple[float, float]]:
    """
    Normalize objectives to [0, 1].

    acc  → higher is better → norm_acc = (acc - acc_min) / (acc_max - acc_min)
    lat  → lower is better  → norm_lat = (lat - lat_min) / (lat_max - lat_min)

    After normalization, we maximize norm_acc and minimize norm_lat.
    """
    pts = []
    for _, acc, lat in front:
        na = (acc - acc_min) / (acc_max - acc_min + 1e-9)
        nl = (lat - lat_min) / (lat_max - lat_min + 1e-9)
        pts.append((na, nl))
    return pts


def _hypervolume_2d(
    front: List[Tuple[np.ndarray, float, float]],
    ref_nacc: float = -0.1,   # reference point: slightly below 0 (worse than worst)
    ref_nlat: float = 1.1,    # reference point: slightly above 1 (worse than worst)
) -> float:
    """
    Normalized 2-D hypervolume indicator.

    Uses normalized objectives in [0,1]:
        norm_acc (maximize), norm_lat (minimize).
    Reference point: (-0.1, 1.1) — matches project convention (1.1, 1.1) in
    the (1-norm_acc, norm_lat) minimization space.

    HV is computed in the space (norm_acc, 1 - norm_lat) with ref (-0.1, -0.1),
    i.e., we convert to a 2-maximization problem then sweep.
    """
    if not front:
        return 0.0
    pts_norm = _normalize_objectives(front)
    # Convert to maximization: (norm_acc, 1 - norm_lat), ref = (-0.1, -0.1)
    pts_max = [(na - ref_nacc, (1.0 - nl) - ref_nacc) for na, nl in pts_norm]
    pts_max = sorted(pts_max, key=lambda t: t[0])
    hv = 0.0
    prev_x = 0.0
    for x, y in pts_max:
        if x > prev_x and y > 0:
            hv += (x - prev_x) * y
            prev_x = x
    return hv


def _igd(
    front: List[Tuple[np.ndarray, float, float]],
    n_ref: int = 50,
) -> float:
    """
    Inverted Generational Distance (IGD) — lower is better.

    Uses a uniformly sampled reference front on the synthetic trade-off curve.
    For each reference point, finds the nearest solution in the found front
    (in normalized objective space). Returns mean distance.
    """
    if not front:
        return float("inf")
    pts_norm = _normalize_objectives(front)
    # Generate reference front: uniform trade-off norm_acc = 1 - norm_lat
    ref_pts = [(t, 1.0 - t) for t in np.linspace(0.0, 1.0, n_ref)]
    total = 0.0
    for rx, ry in ref_pts:
        dists = [math.sqrt((rx - na) ** 2 + (ry - (1.0 - nl)) ** 2)
                 for na, nl in pts_norm]
        total += min(dists)
    return total / n_ref


# ---------------------------------------------------------------------------
# Simulated search loop
# ---------------------------------------------------------------------------

def run_variant(
    operator: GAOperator,
    budget: int,
    seed: int = 42,
    pop_size: int = 20,
    checkpoint_every: int = 100,
    verbose: bool = False,
) -> Dict:
    """
    Run a simplified MOEA/D-style loop using the given GAOperator.

    Returns a dict with:
        hv          — final hypervolume
        pareto_size — number of non-dominated solutions
        clone_rate  — fraction of apply() calls that returned a clone
        mean_hd     — mean Hamming distance (parent → child) over all calls
        hv_curve    — list of (evals, hv) checkpoints
        elapsed     — wall time in seconds
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    # Initialise population
    population = [rng.integers(0, _N_OPS, size=_N_EDGES) for _ in range(pop_size)]
    archive: List[Tuple[np.ndarray, float, float]] = []
    archive_set: Dict[Tuple, Tuple[float, float]] = {}

    def _add_to_archive(arch):
        key = tuple(arch)
        if key not in archive_set:
            acc, lat = _synthetic_objectives(arch)
            archive_set[key] = (acc, lat)
            archive.append((arch.copy(), acc, lat))

    for ind in population:
        _add_to_archive(ind)

    clone_count = 0
    hd_total = 0.0
    total_calls = 0
    hv_curve: List[Tuple[int, float]] = []
    evals = pop_size

    state = MemeticState(population)

    while evals < budget:
        for k in range(pop_size):
            if evals >= budget:
                break

            # Update adaptive schedule
            operator.notify_budget(evals, budget)

            # Update freq-bias stats every 50 evals
            if operator.freq_bias and evals % 50 == 0:
                operator.update_archive_stats([a for a, _, _ in archive])

            parent = state.current[k].copy()
            child = operator.apply(state, k, rng)

            # Track clone rate and Hamming distance
            hd = int(np.sum(child != parent))
            if hd == 0:
                clone_count += 1
            hd_total += hd
            total_calls += 1

            # Accept child if it improves (greedy for simplicity)
            acc_c, lat_c = _synthetic_objectives(child)
            acc_p, lat_p = _synthetic_objectives(parent)
            if acc_c >= acc_p or lat_c <= lat_p:
                state.current[k] = child
            _add_to_archive(child)
            evals += 1

            # Checkpoint
            if evals % checkpoint_every == 0:
                front = _pareto_front(list(archive_set_to_list(archive_set)))
                hv_curve.append((evals, _hypervolume_2d(front)))
                if verbose:
                    print(f"  evals={evals:5d}  HV={hv_curve[-1][1]:.4f}  "
                          f"archive={len(archive_set)}")

    final_front = _pareto_front(list(archive_set_to_list(archive_set)))
    hv  = _hypervolume_2d(final_front)
    igd = _igd(final_front)
    # Final checkpoint
    if not hv_curve or hv_curve[-1][0] != evals:
        hv_curve.append((evals, hv))

    return {
        "hv":           hv,
        "igd":          igd,
        "pareto_size":  len(final_front),
        "clone_rate":   clone_count / max(total_calls, 1),
        "mean_hd":      hd_total / max(total_calls, 1),
        "hv_curve":     hv_curve,
        "elapsed":      time.perf_counter() - t0,
        "archive_size": len(archive_set),
    }


def archive_set_to_list(archive_set):
    """Convert archive dict to (arch, acc, lat) tuples."""
    for key, (acc, lat) in archive_set.items():
        yield (np.array(key), acc, lat)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def run_unit_tests() -> int:
    """Run correctness checks.  Returns number of failures."""
    failures = 0
    rng = np.random.default_rng(0)

    def check(condition: bool, name: str) -> None:
        nonlocal failures
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            failures += 1

    print("\n── Unit tests ──────────────────────────────────────────────────────")

    pop = [rng.integers(0, _N_OPS, _N_EDGES) for _ in range(5)]
    state = MemeticState(pop)

    # T1: Output shape
    op = GAOperator(force_change=False)
    out = op.apply(state, 0, rng)
    check(out.shape == (_N_EDGES,), "Output shape is (6,)")
    check(out.dtype == pop[0].dtype or np.issubdtype(out.dtype, np.integer),
          "Output dtype is integer")

    # T2: All values in valid range
    op2 = GAOperator(force_change=False)
    for _ in range(50):
        out = op2.apply(state, 0, rng)
        check(all(0 <= v < _N_OPS for v in out), "All ops in {0..4}")

    # T3: force_change guarantees Hamming ≥ 1
    op3 = GAOperator(force_change=True)
    all_changed = True
    for k in range(len(pop)):
        for _ in range(30):
            child = op3.apply(state, k, rng)
            if np.array_equal(child, state.current[k]):
                all_changed = False
    check(all_changed, "force_change: no clones produced (30 trials × 5 sub-problems)")

    # T4: Baseline produces some clones (stochastic — run 200 times)
    op4 = GAOperator(force_change=False)
    clone_seen = False
    for _ in range(200):
        child = op4.apply(state, 0, rng)
        if np.array_equal(child, state.current[0]):
            clone_seen = True
            break
    check(clone_seen, "Baseline (force_change=False): clones are possible")

    # T5: Adaptive p_m changes over budget
    op5 = GAOperator(adaptive=True)
    op5.notify_budget(0, 1000)
    pm_start = op5.active_pm
    op5.notify_budget(999, 1000)
    pm_end = op5.active_pm
    check(pm_start > pm_end, f"Adaptive: p_m decreases over budget ({pm_start:.4f} → {pm_end:.4f})")

    # T6: Crossover mixes genes from two parents
    parent_a = np.zeros(_N_EDGES, dtype=int)   # all zeros
    parent_b = np.full(_N_EDGES, 4, dtype=int)  # all fours
    pop2 = [parent_a, parent_b]
    state2 = MemeticState(pop2)
    op6 = GAOperator(crossover=True, force_change=False)
    mixed = False
    for _ in range(50):
        child = op6.apply(state2, 0, rng)
        if 4 in child:
            mixed = True
            break
    check(mixed, "Crossover: child contains genes from both parents")

    # T7: Frequency-bias changes distribution
    op7 = GAOperator(freq_bias=True, force_change=False, p_m=1.0)
    biased_archive = [np.zeros(_N_EDGES, dtype=int)] * 50   # all op-0
    op7.update_archive_stats(biased_archive)
    op7.notify_budget(0, 1000)
    counts_op0 = 0
    for _ in range(200):
        child = op7.apply(state, 0, rng)
        counts_op0 += int(child[0] == 0)
    check(counts_op0 < 100,
          f"Freq-bias: over-represented op-0 selected less often ({counts_op0}/200 times)")

    # T8: variant_name reports flags correctly
    op8 = GAOperator(force_change=True, adaptive=True, crossover=True, freq_bias=True)
    name = op8.variant_name()
    check("force_change" in name and "adaptive_pm" in name
          and "crossover" in name and "freq_bias" in name,
          f"variant_name() reports all flags: {name}")

    # T9: notify_budget clamps at 1.0
    op9 = GAOperator(adaptive=True)
    op9.notify_budget(5000, 1000)
    check(op9._budget_fraction == 1.0, "notify_budget clamps budget_fraction at 1.0")

    print(f"\n  Result: {failures} failure(s)\n")
    return failures


# ---------------------------------------------------------------------------
# Benchmark run
# ---------------------------------------------------------------------------

VARIANTS = [
    ("Baseline (original)",            GAOperator(force_change=False)),
    ("+ force_change",                 GAOperator(force_change=True)),
    ("+ force_change + adaptive_pm",   GAOperator(force_change=True, adaptive=True)),
    ("+ force_change + crossover",     GAOperator(force_change=True, crossover=True)),
    ("All improvements",               GAOperator(force_change=True, adaptive=True,
                                                   crossover=True, freq_bias=True)),
]


def run_benchmark(budget: int = 2000, seed: int = 42) -> List[Dict]:
    print(f"\n── Variant benchmark  (budget={budget}, seed={seed}) ──────────────────")
    results = []
    for name, op in VARIANTS:
        print(f"  Running: {name} ...", end=" ", flush=True)
        r = run_variant(op, budget=budget, seed=seed)
        r["name"] = name
        results.append(r)
        print(f"HV={r['hv']:.4f}  Pareto={r['pareto_size']}  "
              f"clone_rate={r['clone_rate']:.3f}  mean_HD={r['mean_hd']:.3f}  "
              f"({r['elapsed']:.2f}s)")
    return results


def print_comparison_table(results: List[Dict]) -> None:
    """
    Print ablation comparison table using the project's metrics:
      - Normalized HV  (higher is better, ref point (-0.1, 1.1) in norm space)
      - IGD            (lower is better)
      - Pareto size    (number of non-dominated solutions)
      - Clone rate     (fraction of wasted budget evaluations)
      - Mean Hamming distance (perturbation intensity)
    """
    W = 95
    print("\n" + "=" * W)
    print("ABLATION TABLE — GA Operator Variants  |  Metrics: HV (norm.) · IGD · Pareto")
    print("=" * W)
    hdr = (f"{'Variant':<38} {'HV (↑)':>9} {'ΔHVT2':>8} {'IGD (↓)':>9} "
           f"{'Pareto':>7} {'Clone%':>7} {'MeanHD':>7}")
    print(hdr)
    print("─" * W)

    baseline_hv = results[0]["hv"]
    for r in results:
        delta_hv = r["hv"] - baseline_hv
        delta_str = f"{'+' if delta_hv >= 0 else ''}{delta_hv:.4f}" if delta_hv != 0 else "(base)"
        print(
            f"{r['name']:<38} "
            f"{r['hv']:>9.4f} "
            f"{delta_str:>8} "
            f"{r['igd']:>9.4f} "
            f"{r['pareto_size']:>7d} "
            f"{r['clone_rate']*100:>6.1f}% "
            f"{r['mean_hd']:>7.3f}"
        )
    print("─" * W)

    best_hv  = max(results, key=lambda r: r["hv"])
    best_igd = min(results, key=lambda r: r["igd"])
    print(f"  Best HV  → {best_hv['name']}  ({best_hv['hv']:.4f})")
    print(f"  Best IGD → {best_igd['name']}  ({best_igd['igd']:.4f})")
    print("=" * W)
    print("\nMetric notes:")
    print("  HV   — normalized hypervolume; ref point (-0.1, 1.1) in (norm_acc, norm_lat) space")
    print("         matches project convention from src/analysis/pareto_metrics.py")
    print("  IGD  — Inverted Generational Distance vs uniform 50-pt reference front")
    print("  Clone% — % of apply() calls that returned arch identical to parent (wasted budget)")
    print("  MeanHD — mean Hamming distance (parent→child), measures perturbation intensity")


def plot_convergence(results: List[Dict], save_path: str = "ga_convergence.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed — skipping convergence plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")

    colors = ["#64748b", "#38bdf8", "#34d399", "#f97316", "#a78bfa"]
    linestyles = ["--", "-", "-", "-", "-"]

    # ── Left: HV vs evaluations ───────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d1117")
    for i, r in enumerate(results):
        xs = [c[0] for c in r["hv_curve"]]
        ys = [c[1] for c in r["hv_curve"]]
        ax.plot(xs, ys, color=colors[i], lw=1.8,
                linestyle=linestyles[i], label=r["name"])
    ax.set_xlabel("Budget evaluations", color="#94a3b8", fontsize=11)
    ax.set_ylabel("Hypervolume", color="#94a3b8", fontsize=11)
    ax.set_title("Convergence: HV vs evaluations", color="#f1f5f9", fontsize=12)
    ax.tick_params(colors="#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    ax.legend(facecolor="#1e293b", edgecolor="#334155",
              labelcolor="#cbd5e1", fontsize=8)
    ax.grid(True, color="#1e293b", linewidth=0.7)

    # ── Right: final HV bar chart ─────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")
    names_short = [r["name"].replace("+ ", "+\n") for r in results]
    hvs = [r["hv"] for r in results]
    bars = ax2.bar(range(len(results)), hvs, color=colors, width=0.6, zorder=3)
    for bar, val in zip(bars, hvs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + max(hvs) * 0.01,
                 f"{val:.4f}", ha="center", va="bottom",
                 color="#f1f5f9", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(results)))
    ax2.set_xticklabels(names_short, color="#94a3b8", fontsize=8)
    ax2.set_ylabel("Final HV", color="#94a3b8", fontsize=11)
    ax2.set_title("Final hypervolume by variant", color="#f1f5f9", fontsize=12)
    ax2.tick_params(colors="#94a3b8")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#1e293b")
    ax2.set_ylim(0, max(hvs) * 1.15)
    ax2.grid(True, axis="y", color="#1e293b", linewidth=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[Plot] Saved convergence chart: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GA Operator unit tests + variant comparison")
    p.add_argument("--quick",  action="store_true", help="Budget = 500, skip plot")
    p.add_argument("--plot",   action="store_true", help="Save convergence PNG")
    p.add_argument("--budget", type=int, default=2000, help="Eval budget per variant")
    p.add_argument("--seed",   type=int, default=42,   help="RNG seed")
    p.add_argument("--no-tests", action="store_true",  help="Skip unit tests")
    return p.parse_args()


def main():
    args = parse_args()

    budget = 500 if args.quick else args.budget

    print("=" * 60)
    print("GA Operator — Test & Comparison Suite")
    print(f"  Budget per variant : {budget}")
    print(f"  Seed               : {args.seed}")
    print("=" * 60)

    # ── Unit tests ────────────────────────────────────────────────────────
    if not args.no_tests:
        failures = run_unit_tests()
        if failures:
            print(f"\n[WARN] {failures} unit test(s) failed — check your implementation.\n")
    else:
        failures = 0

    # ── Benchmark ─────────────────────────────────────────────────────────
    results = run_benchmark(budget=budget, seed=args.seed)
    print_comparison_table(results)

    # ── Plot ──────────────────────────────────────────────────────────────
    if args.plot and not args.quick:
        plot_convergence(results)
    elif not args.quick:
        print("\n[TIP] Run with --plot to save a convergence chart PNG.")

    # ── Exit code ─────────────────────────────────────────────────────────
    sys.exit(failures)


if __name__ == "__main__":
    main()