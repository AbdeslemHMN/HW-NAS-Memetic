# HW-NAS-Memetic: A Multi-Level Hybrid Metaheuristic for Edge AI

![Python](https://img.shields.io/badge/Python-3.12-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Benchmark](https://img.shields.io/badge/Benchmark-NAS--Bench--201%20%7C%20HW--NAS--Bench-orange)

## 1. Abstract & Motivation

Deploying Deep Learning models on Edge AI devices (smartphones, IoT sensors, UAVs) requires strictly balancing two conflicting objectives: maximizing **Accuracy** while minimizing hardware constraints like **Latency**. Hardware-Aware Neural Architecture Search (HW-NAS) automates this process.

Currently, the State-of-the-Art in multi-objective HW-NAS is dominated by purely Evolutionary Algorithms (e.g., NSGA-II). However, in discrete, cell-based search spaces like NAS-Bench-201, neural architectures are represented as Directed Acyclic Graphs (DAGs) exhibiting extreme **epistasis** — the operations within a network are deeply interdependent. Traditional evolutionary crossover violently breaks these dependencies by slicing and combining disparate graphs, frequently causing catastrophic performance collapse.

**HW-NAS-Memetic** proposes a novel solution: a Multi-Level Hybrid Metaheuristic integrating the global scalarization of **MOEA/D** with the topology-preserving local exploitation of a custom **Discrete Probabilistic PSO** and **Simulated Annealing**, enabling safe per-edge hardware optimization without disrupting the functional graph routing.

## 2. Why "Memetic"?

In optimization literature, there is a strict distinction between Genetic and Memetic algorithms:

- **Genetic Algorithms (e.g., NSGA-II):** Operate strictly on Darwinian evolution. An architecture is born, evaluated, and either breeds or dies. It undergoes no lifetime learning.
- **Memetic Algorithms:** Combine population-based global search with lifelong learning. An architecture is generated globally, then given tools to explore its local neighborhood and actively refine its own structure before committing to the Pareto Front.

For NAS, purely genetic crossover destroys co-adapted data flow. The Discrete PSO component grants each architecture a "lifetime" to consult its personal memory (`pbest`) and swarm memory (`gbest`), tweaking individual edges (e.g., replacing `nor_conv_3x3` with `skip_connect`) to reduce latency while preserving functional routing — a topology-safe local search that evolutionary algorithms structurally lack.

## 3. Key Innovations & Algorithm Architecture

### 3.1 MOEA/D Scalarization

$K$ linear weight vectors decompose the bi-objective space into $K$ scalar sub-problems, each solved independently:

$$g(a \mid w_k) = w_{acc} \cdot Acc(a) - w_{lat} \cdot Lat(a)$$

Weight vectors are linearly spaced from $(w_{acc}, w_{lat}) = (1, 0)$ to $(0, 1)$, uniformly covering the accuracy-latency trade-off spectrum. This reduces Pareto-dominance sorting from $O(N^2)$ to $O(K)$ scalar comparisons per iteration.

Each sub-problem $k$ maintains a neighborhood $\mathcal{B}(k)$ of $T_n$ nearest sub-problems by weight vector Euclidean distance. gbest influence is bounded to $\mathcal{B}(k)$, preventing premature homogenisation across distant trade-off directions.

### 3.2 Dual-Generation Engine

At each iteration, two candidate architectures are generated per sub-problem:

- **GA Mutation** (global perturbation): Per-gene Bernoulli mutation at rate $p_m = 1/L$. Each selected edge is replaced by a uniformly random alternative from $\{0..4\} \setminus \{e_{\text{current}}\}$, ensuring a strict change.
- **Discrete Probabilistic PSO** (local refinement): Each edge $e_i$ is updated according to:

$$P(\text{pull to } p_{\text{best}}) = c_1 \cdot r_1, \quad P(\text{pull to } g_{\text{best}}) = c_2 \cdot r_2$$

where $r_1, r_2 \sim \mathcal{U}(0,1)$. This preserves edges already aligned with high-scoring trajectories, directly addressing the epistasis problem.

The better of the two candidates (by $g(\cdot \mid w_k)$) proceeds to the acceptance step.

### 3.3 Simulated Annealing Trajectory Control

The Metropolis criterion governs acceptance of the selected candidate:

$$P(\text{accept}) = \begin{cases} 1 & \text{if } \Delta g \geq 0 \\ \exp\!\left(\Delta g / T\right) & \text{otherwise} \end{cases}$$

where $\Delta g = g_{\text{candidate}} - g_{\text{current}}$. Temperature decays geometrically: $T_{t+1} = \alpha \cdot T_t$, $\alpha \in (0,1)$. High $T$ permits diversity-preserving uphill moves; low $T$ collapses to greedy exploitation.

### 3.4 A Posteriori Pareto Filtering

The global archive records every evaluated architecture. The final Pareto front is extracted post-search via non-dominated sorting on $(-Acc, Lat)$, decoupling the search trajectory from the output representation.

## 4. Search Space & Problem Encoding

### NAS-Bench-201 Cell

The search space is a 4-node DAG with $L = 6$ directed edges (all node pairs $(i \to j)$ for $i < j$, $i,j \in \{0,1,2,3\}$). Each edge independently selects one of $O = 5$ operations:

| Index | Operation |
|-------|-----------|
| 0 | `none` (zero tensor) |
| 1 | `skip_connect` (identity) |
| 2 | `avg_pool_3x3` |
| 3 | `nor_conv_1x1` |
| 4 | `nor_conv_3x3` |

Total search space: $5^6 = 15{,}625$ architectures.

### Encoding

An architecture is encoded as an integer vector $a \in \{0,1,2,3,4\}^6$. All operators enforce `shape == (6,)` and `values in [0, 4]`.

### Hardware Evaluation

HW-NAS-Bench provides pre-computed hardware metrics for all 15,625 architectures across 7 devices on 3 datasets:

- **Devices**: `edgegpu`, `raspi4`, `edgetpu`, `pixel3`, `eyeriss`, `fpga`
- **Metrics**: latency (ms), energy (mJ), arithmetic intensity
- **Datasets**: `cifar10`, `cifar100`, `ImageNet16-120`

Lookup is $O(1)$ via a pre-built `arch_tuple → index` dictionary in `src/api/hw_nas_wrapper.py`.

## 5. Repository Structure

```
HW-NAS-Memetic/
├── data/
│   └── HW-NAS-Bench-v1_0.pickle      # Hardware benchmark (download separately)
├── notebooks/
│   ├── 01_pareto_front_viz.ipynb      # Pareto front scatter + HV/IGD table
│   └── 02_epistasis_heatmap.ipynb     # Edge perturbation brittleness heatmap
├── results/
│   ├── baselines/                     # RandomSearch and NSGA-II JSON archives
│   └── proposed/                      # ProposedMoeadPso JSON archives
├── scripts/
│   ├── download_data.py               # Automated dataset download
│   ├── run_baselines.py               # 5-seed baseline sweep
│   └── run_search.py                  # Proposed algorithm CLI
├── src/
│   ├── algorithms/
│   │   ├── base_optimizer.py          # Abstract BaseOptimizer
│   │   ├── random_search.py           # Random Search baseline
│   │   ├── proposed_moead_pso.py      # MOEA/D + Discrete PSO + SA (proposed)
│   │   └── nsga2_search.py            # NSGA-II baseline (pymoo)
│   ├── analysis/
│   │   └── pareto_metrics.py          # HV, IGD, proxy Pareto utilities
│   ├── api/
│   │   └── hw_nas_wrapper.py          # O(1) HW-NAS-Bench query interface
│   ├── operators/
│   │   ├── discrete_ga.py             # Mutation and crossover
│   │   ├── discrete_pso.py            # Probabilistic discrete PSO step
│   │   └── simulated_annealing.py     # Metropolis acceptance and cooling
│   └── utils/
│       └── logger.py                  # save_archive (JSON/CSV) + get_logger
└── test/
    └── test_data_loading.py           # Unit tests (9 tests, data + API)
```

## 6. Installation & Environment Setup

**Requirements**: Python 3.12, pip.

```bash
git clone https://github.com/AbdeslemHMN/HW-NAS-Memetic.git
cd HW-NAS-Memetic
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Dataset**: Download `HW-NAS-Bench-v1_0.pickle` and place it in `data/`:

```bash
python scripts/download_data.py
```

**Optional: build the NAS-Bench-201 accuracy cache**

If you want ground-truth NAS-Bench-201 accuracy instead of the structural proxy, run:

```bash
python scripts/extract_nas201_accuracy.py
```

This script now queries NATS-Bench TSS in a memory-safe way by evicting each architecture from the internal cache immediately after use, avoiding the out-of-memory / system freeze issue.

Alternatively, download manually from the [HW-NAS-Bench release](https://github.com/RICE-EIC/HW-NAS-Bench) and place the file at `data/HW-NAS-Bench-v1_0.pickle`.

**Verify installation:**

```bash
python3 -m unittest test.test_data_loading
```

Expected: `Ran 9 tests in ~1s OK`.

## 7. Quickstart & Usage

**Step 1 — Clone and install** (see Section 6).

**Step 2 — Place benchmark data** at `data/HW-NAS-Bench-v1_0.pickle`.

**Step 3 — Run baselines** (30 seeds, RandomSearch + NSGA-II):

```bash
python scripts/run_baselines.py
```

Outputs saved to `results/baselines/random_search_seed{N}.json` and `results/baselines/nsga2_seed{N}.json`.

**Step 4 — Run proposed search:**

```bash
python scripts/run_search.py \
    --hardware edgegpu_latency \
    --budget 2000 \
    --k_directions 20
```

All CLI flags with defaults:

| Flag | Default | Description |
|------|---------|-------------|
| `--hardware` | `edgegpu_latency` | Hardware metric key |
| `--budget` | `2000` | Total architecture evaluations |
| `--k_directions` | `20` | MOEA/D weight vectors $K$ |
| `--t_neighborhood` | `2` | Neighborhood size $T_n$ |
| `--t0` | `1.0` | SA initial temperature |
| `--alpha` | `0.95` | SA cooling rate |
| `--c1` | `0.5` | PSO personal attraction |
| `--c2` | `0.5` | PSO global attraction |
| `--dataset` | `cifar10` | NAS-Bench-201 dataset split |
| `--seed` | `42` | Random seed |

Output saved to `results/proposed/proposed_res_{seed}.json`.

**Step 5 — Visualize results** via Jupyter:

```bash
jupyter notebook notebooks/
```

- `01_pareto_front_viz.ipynb` — Pareto front scatter plot + HV/IGD comparison table.
- `02_epistasis_heatmap.ipynb` — Per-edge brittleness heatmap over top-10 architectures.

## 8. Baselines & Evaluation Metrics

Two baselines are implemented for rigorous comparison:

| Algorithm | Module | Strategy |
|-----------|--------|----------|
| Random Search | `src/algorithms/random_search.py` | i.i.d. uniform sampling over $\{0..4\}^6$ |
| NSGA-II | `src/algorithms/nsga2_search.py` | pymoo 0.6 with integer SBX + PM + RoundingRepair |

### Metrics

All metrics computed in `src/analysis/pareto_metrics.py` under pymoo's minimisation convention $F = (-Acc, Lat)$:

**Hypervolume (HV):** Volume of objective space dominated by the Pareto front, bounded by reference point $r$:

$$HV(F, r) = \lambda\left(\bigcup_{f \in F} [f, r]\right)$$

Higher is better. Points are normalised to $[0,1]^2$ before cross-algorithm comparison using the proxy Pareto range as scale.

**Inverted Generational Distance (IGD):** Mean distance from the proxy reference front $P^{\ast}$ to the nearest solution in the approximation set $F$:

$$IGD(F, P^{\ast}) = \frac{1}{|P^{\ast}|} \sum_{p \in P^{\ast}} \min_{f \in F} \lVert p - f \rVert_2$$

Lower is better. $P^{\ast}$ is constructed as the non-dominated union of all algorithm runs across all seeds.

## 9. Implemented Upgrades (Advanced Features)

The repository now enforces a strict, budget-fair comparison across all algorithms. The key upgrades are:

- **30 independent seeds for every algorithm**
  - `scripts/run_baselines.py` runs `RandomSearch` and `NSGA-II` for 30 seeds.
  - `scripts/run_search.py` now runs the proposed algorithm for 30 seeds and writes outputs to `results/proposed/proposed_res_seed{seed}.json`.

- **Exact NFE budget enforcement**
  - `src/algorithms/base_optimizer.py` tracks `budget_spent` as the exact Number of Function Evaluations (NFE).
  - `_eval()` raises immediately if `budget_spent >= budget`, guaranteeing no algorithm can exceed the requested 2000 evaluations.

- **Convergence trajectory snapshots**
  - Every 100 NFEs, the optimizer records a checkpoint in `nfe_checkpoints`.
  - This enables convergence plots of Hypervolume versus NFE, rather than only final performance.

- **Fair NSGA-II hyperparameters**
  - `src/algorithms/nsga2_search.py` now uses standard SBX/PM values:
    - `crossover prob = 0.9`, `eta=20`
    - `mutation prob = 1/n_var`, `eta=20`
  - Termination is now `('n_eval', budget)`, matching the proposed algorithm's strict NFE cap.

- **Statistical hypothesis testing**
  - The comparison notebooks now include Wilcoxon Rank-Sum tests for HV and IGD.
  - Bonferroni correction is applied to control familywise error across multiple metric comparisons.

## 10. Recommended Run Sequence

Use this exact order for fair evaluation:

```bash
python scripts/run_baselines.py
python scripts/run_search.py --hardware edgegpu_latency --budget 2000 --k_directions 20
```

Then open the notebooks for analysis:

```bash
jupyter notebook notebooks/
```

Recommended notebooks:

- `notebooks/01_pareto_front_viz.ipynb`
- `notebooks/03_algorithm_comparison.ipynb`

## 11. Expected Output Files

- `results/baselines/random_search_seed{0..4}.json`
- `results/baselines/nsga2_seed{0..4}.json`
- `results/proposed/proposed_res_seed{0..4}.json`
- `results/convergence_trajectory.pdf`
- `results/ablation_pareto_overlay.pdf`
- `results/ablations_hv_boxplot.pdf`


### 9.1 Self-Adaptive PSO Parameters

Rather than fixed $c_1, c_2$, coefficients adapt per sub-problem $k$ based on recent acceptance rate:

$$c_1^{(t+1)} = c_1^{(t)} + \eta \cdot (\rho_{\text{pbest}} - \rho^{\ast}), \quad c_2^{(t+1)} = c_2^{(t)} + \eta \cdot (\rho_{\text{gbest}} - \rho^{\ast})$$

where $\rho^{\ast}$ is a target success rate and $\eta$ is the adaptation step. Sub-problems stuck in local optima automatically increase $c_2$ (exploitation), while diverse sub-problems increase $c_1$ (exploration).

### 9.2 Multi-Armed Bandit (MAB) Tool Selection

Instead of always generating both GA and PSO candidates (2 evaluations per sub-problem), a $\varepsilon$-greedy bandit selects the single operator with the highest empirical improvement rate. This halves the evaluation budget consumed per cycle when one operator consistently dominates, concentrating budget in the productive search phase.

### 9.3 Dynamic Weight Adaptation

Weight vectors $w_k$ are periodically redistributed toward the region of highest crowding distance on the current Pareto approximation. Sparse regions attract more vectors, preventing over-concentration of solutions in the accuracy-dominant or latency-dominant extremes and improving Hypervolume coverage.

### 9.4 Block-wise PSO

Rather than per-edge probabilistic updates, edges are grouped into structural blocks (e.g., input edges $\{0,1\}$, middle edges $\{2,3\}$, output edges $\{4,5\}$). PSO updates are applied block-wise, preserving intra-block co-adaptation and reducing the probability of disrupting functionally coupled edge groups — directly targeting the epistasis problem at its structural source.

## 10. Citation & Acknowledgements

If you use this repository in your research, please cite:

```bibtex
@misc{hwnas_memetic_2026,
  author       = {Abdeslem, HMN},
  title        = {HW-NAS-Memetic: A Multi-Level Hybrid Metaheuristic for Hardware-Aware Neural Architecture Search},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/AbdeslemHMN/HW-NAS-Memetic}
}
```

This work builds on the following benchmarks:

- **NAS-Bench-201**: Dong, X. & Yang, Y. (2020). NAS-Bench-201: Extending the Scope of Reproducible Neural Architecture Search. *ICLR 2020*. [[Paper](https://arxiv.org/abs/2001.00326)]
- **HW-NAS-Bench**: Li, C. et al. (2021). HW-NAS-Bench: Hardware-Aware Neural Architecture Search Benchmark. *ICLR 2021*. [[Paper](https://arxiv.org/abs/2103.10584)]
- **pymoo**: Blank, J. & Deb, K. (2020). pymoo: Multi-Objective Optimization in Python. *IEEE Access*. [[Paper](https://ieeexplore.ieee.org/document/9078759)]