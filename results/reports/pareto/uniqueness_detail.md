# Architectural Genotype Uniqueness — Detailed Report
_Generated: 2026-05-14 00:19 UTC_


---

## cifar10 / Edge GPU

- **Proposed front size**: 27 distinct genotypes
- **Global novelty**: 0.0% (0 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 28 |
| Shared architectures | 27 |
| Unique to Proposed | 0 |
| Unique to NSGA-II | 1 |
| Novelty score (Proposed vs NSGA-II) | 0.0% |

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 28 |
| Shared architectures | 25 |
| Unique to Proposed | 2 |
| Unique to Random Search | 3 |
| Novelty score (Proposed vs Random Search) | 7.4% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[4, 1, 1, 1, 4, 0]`
- `[4, 4, 2, 4, 0, 0]`
</details>

---

## cifar10 / Raspi4

- **Proposed front size**: 45 distinct genotypes
- **Global novelty**: 24.4% (11 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 43 |
| Shared architectures | 34 |
| Unique to Proposed | 11 |
| Unique to NSGA-II | 9 |
| Novelty score (Proposed vs NSGA-II) | 24.4% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[0, 0, 1, 4, 0, 0]`
- `[1, 1, 0, 0, 0, 4]`
- `[1, 4, 0, 1, 0, 3]`
- `[1, 4, 3, 1, 0, 3]`
- `[3, 0, 2, 4, 0, 0]`
- `[3, 3, 2, 1, 1, 3]`
- `[3, 3, 4, 1, 0, 3]`
- `[3, 4, 2, 1, 3, 4]`
- `[3, 4, 4, 1, 4, 4]`
- `[4, 0, 4, 1, 4, 4]`
- `[4, 4, 2, 1, 3, 3]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 42 |
| Shared architectures | 32 |
| Unique to Proposed | 13 |
| Unique to Random Search | 10 |
| Novelty score (Proposed vs Random Search) | 28.9% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[0, 0, 1, 4, 0, 0]`
- `[1, 1, 0, 0, 0, 4]`
- `[1, 4, 0, 1, 0, 3]`
- `[1, 4, 3, 1, 0, 3]`
- `[3, 0, 2, 4, 0, 0]`
- `[3, 3, 2, 1, 1, 3]`
- `[3, 3, 4, 1, 0, 3]`
- `[3, 4, 2, 1, 3, 4]`
- `[3, 4, 4, 1, 4, 4]`
- `[4, 0, 4, 1, 4, 4]`
- `[4, 2, 3, 1, 4, 0]`
- `[4, 4, 2, 1, 3, 3]`
- `[4, 4, 2, 4, 0, 0]`
</details>

---

## cifar10 / Eyeriss

- **Proposed front size**: 343 distinct genotypes
- **Global novelty**: 0.0% (0 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 366 |
| Shared architectures | 342 |
| Unique to Proposed | 1 |
| Unique to NSGA-II | 24 |
| Novelty score (Proposed vs NSGA-II) | 0.3% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[1, 1, 1, 0, 4, 0]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 360 |
| Shared architectures | 336 |
| Unique to Proposed | 7 |
| Unique to Random Search | 24 |
| Novelty score (Proposed vs Random Search) | 2.0% |

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

- **Proposed front size**: 29 distinct genotypes
- **Global novelty**: 31.0% (9 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 25 |
| Shared architectures | 20 |
| Unique to Proposed | 9 |
| Unique to NSGA-II | 5 |
| Novelty score (Proposed vs NSGA-II) | 31.0% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[1, 2, 4, 1, 0, 0]`
- `[1, 3, 1, 0, 0, 0]`
- `[1, 3, 3, 1, 0, 0]`
- `[1, 4, 0, 3, 0, 0]`
- `[3, 0, 4, 1, 0, 0]`
- `[3, 3, 2, 1, 0, 0]`
- `[4, 1, 4, 1, 0, 0]`
- `[4, 2, 1, 1, 0, 0]`
- `[4, 3, 4, 1, 0, 0]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 27 |
| Shared architectures | 18 |
| Unique to Proposed | 11 |
| Unique to Random Search | 9 |
| Novelty score (Proposed vs Random Search) | 37.9% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[1, 2, 4, 1, 0, 0]`
- `[1, 3, 0, 1, 0, 0]`
- `[1, 3, 1, 0, 0, 0]`
- `[1, 3, 3, 1, 0, 0]`
- `[1, 4, 0, 3, 0, 0]`
- `[3, 0, 4, 1, 0, 0]`
- `[3, 3, 2, 1, 0, 0]`
- `[4, 1, 4, 1, 0, 0]`
- `[4, 2, 1, 1, 0, 0]`
- `[4, 3, 4, 1, 0, 0]`
- `[4, 4, 2, 4, 0, 0]`
</details>

---

## cifar100 / Raspi4

- **Proposed front size**: 38 distinct genotypes
- **Global novelty**: 15.8% (6 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 39 |
| Shared architectures | 32 |
| Unique to Proposed | 6 |
| Unique to NSGA-II | 7 |
| Novelty score (Proposed vs NSGA-II) | 15.8% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[0, 4, 1, 1, 1, 3]`
- `[1, 0, 0, 1, 0, 1]`
- `[3, 2, 1, 1, 0, 0]`
- `[4, 3, 0, 1, 3, 3]`
- `[4, 3, 3, 1, 3, 3]`
- `[4, 3, 4, 1, 3, 3]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 39 |
| Shared architectures | 32 |
| Unique to Proposed | 6 |
| Unique to Random Search | 7 |
| Novelty score (Proposed vs Random Search) | 15.8% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[0, 4, 1, 1, 1, 3]`
- `[1, 0, 0, 1, 0, 1]`
- `[3, 2, 1, 1, 0, 0]`
- `[4, 3, 0, 1, 3, 3]`
- `[4, 3, 3, 1, 3, 3]`
- `[4, 3, 4, 1, 3, 3]`
</details>

---

## cifar100 / Eyeriss

- **Proposed front size**: 201 distinct genotypes
- **Global novelty**: 4.0% (8 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 299 |
| Shared architectures | 192 |
| Unique to Proposed | 9 |
| Unique to NSGA-II | 107 |
| Novelty score (Proposed vs NSGA-II) | 4.5% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[0, 0, 0, 3, 0, 0]`
- `[0, 1, 1, 1, 3, 3]`
- `[0, 1, 3, 0, 1, 1]`
- `[1, 0, 1, 1, 3, 3]`
- `[1, 0, 1, 3, 1, 3]`
- `[1, 3, 0, 1, 1, 3]`
- `[1, 3, 1, 0, 1, 0]`
- `[3, 1, 0, 1, 0, 0]`
- `[3, 1, 3, 1, 1, 0]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 296 |
| Shared architectures | 190 |
| Unique to Proposed | 11 |
| Unique to Random Search | 106 |
| Novelty score (Proposed vs Random Search) | 5.5% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[0, 1, 1, 1, 3, 3]`
- `[0, 1, 3, 0, 1, 1]`
- `[1, 0, 1, 1, 3, 3]`
- `[1, 0, 1, 3, 1, 3]`
- `[1, 3, 0, 1, 1, 3]`
- `[1, 3, 1, 0, 1, 0]`
- `[3, 0, 1, 1, 0, 4]`
- `[3, 0, 3, 1, 0, 1]`
- `[3, 1, 0, 1, 0, 0]`
- `[3, 1, 3, 1, 1, 0]`
- `[4, 0, 3, 1, 1, 1]`
</details>

---

## ImageNet16-120 / Edge GPU

- **Proposed front size**: 30 distinct genotypes
- **Global novelty**: 6.7% (2 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 35 |
| Shared architectures | 27 |
| Unique to Proposed | 3 |
| Unique to NSGA-II | 8 |
| Novelty score (Proposed vs NSGA-II) | 10.0% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[1, 4, 4, 4, 0, 0]`
- `[2, 0, 0, 4, 0, 3]`
- `[4, 0, 4, 1, 0, 0]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 36 |
| Shared architectures | 25 |
| Unique to Proposed | 5 |
| Unique to Random Search | 11 |
| Novelty score (Proposed vs Random Search) | 16.7% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[1, 4, 4, 4, 0, 0]`
- `[4, 0, 4, 1, 0, 0]`
- `[4, 0, 4, 1, 3, 4]`
- `[4, 2, 3, 1, 4, 0]`
- `[4, 3, 4, 4, 0, 0]`
</details>

---

## ImageNet16-120 / Raspi4

- **Proposed front size**: 50 distinct genotypes
- **Global novelty**: 8.0% (4 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 53 |
| Shared architectures | 44 |
| Unique to Proposed | 6 |
| Unique to NSGA-II | 9 |
| Novelty score (Proposed vs NSGA-II) | 12.0% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[0, 0, 0, 0, 0, 1]`
- `[1, 1, 0, 0, 1, 4]`
- `[3, 1, 3, 4, 0, 0]`
- `[4, 1, 1, 1, 0, 0]`
- `[4, 1, 2, 4, 0, 0]`
- `[4, 1, 4, 4, 0, 1]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 55 |
| Shared architectures | 46 |
| Unique to Proposed | 4 |
| Unique to Random Search | 9 |
| Novelty score (Proposed vs Random Search) | 8.0% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[0, 0, 0, 0, 0, 1]`
- `[3, 1, 3, 4, 0, 0]`
- `[4, 1, 1, 1, 0, 0]`
- `[4, 1, 2, 4, 0, 0]`
</details>

---

## ImageNet16-120 / Eyeriss

- **Proposed front size**: 101 distinct genotypes
- **Global novelty**: 17.8% (18 archs missed by ALL baselines)

### vs NSGA-II
| Metric | Value |
|--------|-------|
| Baseline front size | 315 |
| Shared architectures | 82 |
| Unique to Proposed | 19 |
| Unique to NSGA-II | 233 |
| Novelty score (Proposed vs NSGA-II) | 18.8% |

<details><summary>Architectures unique to Proposed (vs NSGA-II)</summary>

- `[0, 0, 0, 1, 3, 0]`
- `[0, 1, 3, 1, 0, 3]`
- `[0, 3, 0, 1, 3, 3]`
- `[0, 3, 1, 1, 3, 3]`
- `[0, 3, 3, 1, 1, 3]`
- `[1, 0, 1, 3, 1, 3]`
- `[1, 3, 0, 1, 3, 3]`
- `[3, 0, 0, 1, 1, 3]`
- `[3, 0, 0, 1, 3, 3]`
- `[3, 0, 0, 3, 4, 1]`
- `[3, 1, 0, 1, 0, 0]`
- `[3, 1, 1, 0, 1, 3]`
- `[3, 1, 3, 1, 3, 0]`
- `[3, 1, 3, 3, 0, 1]`
- `[3, 3, 0, 1, 0, 3]`
- `[3, 3, 0, 1, 1, 3]`
- `[3, 3, 1, 1, 0, 3]`
- `[3, 3, 3, 1, 0, 1]`
- `[3, 3, 3, 1, 1, 1]`
</details>

### vs Random Search
| Metric | Value |
|--------|-------|
| Baseline front size | 317 |
| Shared architectures | 81 |
| Unique to Proposed | 20 |
| Unique to Random Search | 236 |
| Novelty score (Proposed vs Random Search) | 19.8% |

<details><summary>Architectures unique to Proposed (vs Random Search)</summary>

- `[0, 0, 0, 1, 3, 0]`
- `[0, 1, 3, 1, 0, 3]`
- `[0, 3, 0, 1, 3, 3]`
- `[0, 3, 1, 1, 3, 3]`
- `[0, 3, 3, 1, 1, 3]`
- `[1, 0, 1, 3, 1, 3]`
- `[1, 3, 0, 0, 3, 4]`
- `[1, 3, 0, 1, 3, 3]`
- `[3, 0, 0, 1, 1, 3]`
- `[3, 0, 0, 1, 3, 3]`
- `[3, 1, 0, 1, 0, 0]`
- `[3, 1, 1, 0, 1, 3]`
- `[3, 1, 3, 1, 3, 0]`
- `[3, 1, 3, 3, 0, 1]`
- `[3, 3, 0, 1, 0, 3]`
- `[3, 3, 0, 1, 1, 3]`
- `[3, 3, 1, 1, 0, 3]`
- `[3, 3, 3, 1, 0, 1]`
- `[3, 3, 3, 1, 1, 1]`
- `[4, 0, 4, 1, 3, 4]`
</details>
