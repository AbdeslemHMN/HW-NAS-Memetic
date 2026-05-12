# Architectural Genotype Uniqueness — Detailed Report
_Generated: 2026-05-11 21:01 UTC_


---

## cifar10 / Edge GPU

- **Proposed front size**: 28 distinct genotypes
- **Global novelty**: 0.0% (0 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 28 |
| Shared architectures | 28 |
| Unique to Proposed | 0 |
| Unique to NSGA-II | 0 |
| Novelty score (Proposed vs NSGA-II) | 0.0% |

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 28 |
| Shared architectures | 26 |
| Unique to Proposed | 2 |
| Unique to Random Search | 2 |
| Novelty score (Proposed vs Random Search) | 7.1% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[4, 1, 1, 1, 4, 0]`
- `[4, 4, 2, 4, 0, 0]`
</details>

---

## cifar10 / Raspi4

- **Proposed front size**: 45 distinct genotypes
- **Global novelty**: 11.1% (5 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 43 |
| Shared architectures | 40 |
| Unique to Proposed | 5 |
| Unique to NSGA-II | 3 |
| Novelty score (Proposed vs NSGA-II) | 11.1% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[1, 4, 0, 3, 1, 3]`
- `[3, 3, 4, 1, 0, 4]`
- `[4, 4, 0, 1, 3, 4]`
- `[4, 4, 2, 1, 4, 3]`
- `[4, 4, 4, 1, 3, 3]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 42 |
| Shared architectures | 38 |
| Unique to Proposed | 7 |
| Unique to Random Search | 4 |
| Novelty score (Proposed vs Random Search) | 15.6% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[1, 4, 0, 3, 1, 3]`
- `[3, 3, 4, 1, 0, 4]`
- `[4, 2, 3, 1, 4, 0]`
- `[4, 4, 0, 1, 3, 4]`
- `[4, 4, 2, 1, 4, 3]`
- `[4, 4, 2, 4, 0, 0]`
- `[4, 4, 4, 1, 3, 3]`
</details>

---

## cifar10 / Eyeriss

- **Proposed front size**: 365 distinct genotypes
- **Global novelty**: 0.0% (0 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 366 |
| Shared architectures | 364 |
| Unique to Proposed | 1 |
| Unique to NSGA-II | 2 |
| Novelty score (Proposed vs NSGA-II) | 0.3% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[1, 1, 1, 0, 4, 0]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 360 |
| Shared architectures | 358 |
| Unique to Proposed | 7 |
| Unique to Random Search | 2 |
| Novelty score (Proposed vs Random Search) | 1.9% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[0, 1, 1, 0, 0, 1]`
- `[1, 0, 1, 1, 0, 0]`
- `[1, 1, 0, 1, 3, 1]`
- `[1, 1, 0, 3, 0, 0]`
- `[3, 0, 3, 1, 0, 1]`
- `[4, 0, 3, 1, 1, 1]`
- `[4, 1, 1, 0, 1, 0]`
</details>

---

## cifar100 / Edge GPU

- **Proposed front size**: 24 distinct genotypes
- **Global novelty**: 4.2% (1 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 25 |
| Shared architectures | 23 |
| Unique to Proposed | 1 |
| Unique to NSGA-II | 2 |
| Novelty score (Proposed vs NSGA-II) | 4.2% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[1, 4, 0, 3, 0, 0]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 27 |
| Shared architectures | 21 |
| Unique to Proposed | 3 |
| Unique to Random Search | 6 |
| Novelty score (Proposed vs Random Search) | 12.5% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[1, 3, 0, 1, 0, 0]`
- `[1, 4, 0, 3, 0, 0]`
- `[4, 4, 2, 4, 0, 0]`
</details>

---

## cifar100 / Raspi4

- **Proposed front size**: 39 distinct genotypes
- **Global novelty**: 0.0% (0 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 39 |
| Shared architectures | 39 |
| Unique to Proposed | 0 |
| Unique to NSGA-II | 0 |
| Novelty score (Proposed vs NSGA-II) | 0.0% |

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 39 |
| Shared architectures | 39 |
| Unique to Proposed | 0 |
| Unique to Random Search | 0 |
| Novelty score (Proposed vs Random Search) | 0.0% |

---

## cifar100 / Eyeriss

- **Proposed front size**: 298 distinct genotypes
- **Global novelty**: 0.0% (0 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 299 |
| Shared architectures | 294 |
| Unique to Proposed | 4 |
| Unique to NSGA-II | 5 |
| Novelty score (Proposed vs NSGA-II) | 1.3% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[0, 0, 0, 3, 0, 0]`
- `[0, 0, 0, 3, 0, 1]`
- `[0, 1, 0, 3, 0, 0]`
- `[0, 1, 1, 3, 1, 0]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 296 |
| Shared architectures | 291 |
| Unique to Proposed | 7 |
| Unique to Random Search | 5 |
| Novelty score (Proposed vs Random Search) | 2.3% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[0, 1, 1, 0, 0, 1]`
- `[1, 0, 1, 1, 0, 0]`
- `[1, 1, 0, 1, 3, 1]`
- `[1, 1, 0, 3, 0, 0]`
- `[3, 0, 1, 1, 0, 4]`
- `[3, 0, 3, 1, 0, 1]`
- `[4, 0, 3, 1, 1, 1]`
</details>

---

## ImageNet16-120 / Edge GPU

- **Proposed front size**: 36 distinct genotypes
- **Global novelty**: 0.0% (0 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 35 |
| Shared architectures | 35 |
| Unique to Proposed | 1 |
| Unique to NSGA-II | 0 |
| Novelty score (Proposed vs NSGA-II) | 2.8% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[2, 0, 0, 4, 0, 3]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 36 |
| Shared architectures | 33 |
| Unique to Proposed | 3 |
| Unique to Random Search | 3 |
| Novelty score (Proposed vs Random Search) | 8.3% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[4, 0, 4, 1, 3, 4]`
- `[4, 2, 3, 1, 4, 0]`
- `[4, 3, 4, 4, 0, 0]`
</details>

---

## ImageNet16-120 / Raspi4

- **Proposed front size**: 52 distinct genotypes
- **Global novelty**: 1.9% (1 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 53 |
| Shared architectures | 46 |
| Unique to Proposed | 6 |
| Unique to NSGA-II | 7 |
| Novelty score (Proposed vs NSGA-II) | 11.5% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[1, 1, 0, 0, 1, 4]`
- `[2, 1, 0, 4, 0, 0]`
- `[2, 1, 4, 4, 0, 0]`
- `[3, 4, 0, 3, 0, 0]`
- `[4, 1, 4, 4, 0, 1]`
- `[4, 2, 0, 4, 0, 0]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 55 |
| Shared architectures | 51 |
| Unique to Proposed | 1 |
| Unique to Random Search | 4 |
| Novelty score (Proposed vs Random Search) | 1.9% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[3, 4, 0, 3, 0, 0]`
</details>

---

## ImageNet16-120 / Eyeriss

- **Proposed front size**: 279 distinct genotypes
- **Global novelty**: 4.7% (13 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 315 |
| Shared architectures | 257 |
| Unique to Proposed | 22 |
| Unique to NSGA-II | 58 |
| Novelty score (Proposed vs NSGA-II) | 7.9% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[0, 0, 0, 3, 0, 0]`
- `[0, 0, 1, 3, 0, 0]`
- `[0, 1, 0, 4, 0, 3]`
- `[0, 1, 1, 1, 3, 3]`
- `[0, 1, 3, 1, 0, 3]`
- `[0, 1, 3, 1, 1, 3]`
- `[0, 1, 3, 3, 1, 1]`
- `[0, 4, 0, 3, 0, 1]`
- `[1, 0, 0, 1, 3, 3]`
- `[1, 0, 0, 4, 3, 1]`
- `[1, 0, 3, 3, 1, 4]`
- `[1, 1, 0, 3, 1, 3]`
- `[1, 1, 0, 4, 0, 3]`
- `[1, 1, 1, 1, 3, 3]`
- `[1, 1, 3, 0, 3, 1]`
- `[1, 4, 1, 0, 0, 3]`
- `[3, 0, 0, 1, 1, 3]`
- `[3, 0, 0, 3, 4, 1]`
- `[3, 1, 0, 1, 1, 3]`
- `[3, 1, 1, 0, 1, 3]`
- `[3, 1, 1, 1, 1, 3]`
- `[3, 3, 1, 1, 0, 1]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 317 |
| Shared architectures | 259 |
| Unique to Proposed | 20 |
| Unique to Random Search | 58 |
| Novelty score (Proposed vs Random Search) | 7.2% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[0, 1, 1, 0, 0, 1]`
- `[0, 1, 1, 1, 3, 3]`
- `[0, 1, 3, 1, 0, 3]`
- `[0, 1, 3, 1, 1, 3]`
- `[0, 1, 3, 3, 1, 1]`
- `[1, 0, 0, 1, 3, 3]`
- `[1, 0, 1, 1, 0, 0]`
- `[1, 1, 0, 3, 0, 0]`
- `[1, 1, 0, 3, 1, 3]`
- `[1, 1, 1, 1, 3, 3]`
- `[1, 1, 3, 0, 3, 1]`
- `[1, 3, 0, 0, 3, 4]`
- `[3, 0, 0, 1, 1, 3]`
- `[3, 0, 3, 1, 0, 1]`
- `[3, 1, 0, 1, 1, 3]`
- `[3, 1, 1, 0, 1, 3]`
- `[3, 1, 1, 1, 1, 3]`
- `[3, 3, 1, 1, 0, 1]`
- `[4, 0, 3, 1, 1, 1]`
- `[4, 0, 4, 1, 3, 4]`
</details>
