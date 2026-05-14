# Architectural Genotype Uniqueness Summary
_Generated: 2026-05-14 00:19 UTC_

## Overview

This report measures how many distinct architecture strings (genotypes)
appear on the Pareto front of each algorithm, and how many of the
Proposed algorithm's Pareto-optimal architectures were **not** discovered
by any baseline.

**Novelty score** = (unique-to-proposed / proposed front size) × 100%

## Summary Table

| Dataset        | Hardware | Proposed Front | Global Novelty% | NSGA-II Front | Random Search Front | Shared vs NSGA-II | Shared vs Random Search | Unique vs NSGA-II | Unique vs Random Search | Fully Unique (all BLs) |
| -------------- | -------- | -------------- | --------------- | ------------- | ------------------- | ----------------- | ----------------------- | ----------------- | ----------------------- | ---------------------- |
| cifar10        | Edge GPU | 27             | 0.0%            | 28            | 28                  | 27                | 25                      | 0                 | 2                       | 0                      |
| cifar10        | Raspi4   | 45             | 24.4%           | 43            | 42                  | 34                | 32                      | 11                | 13                      | 11                     |
| cifar10        | Eyeriss  | 343            | 0.0%            | 366           | 360                 | 342               | 336                     | 1                 | 7                       | 0                      |
| cifar100       | Edge GPU | 29             | 31.0%           | 25            | 27                  | 20                | 18                      | 9                 | 11                      | 9                      |
| cifar100       | Raspi4   | 38             | 15.8%           | 39            | 39                  | 32                | 32                      | 6                 | 6                       | 6                      |
| cifar100       | Eyeriss  | 201            | 4.0%            | 299           | 296                 | 192               | 190                     | 9                 | 11                      | 8                      |
| ImageNet16-120 | Edge GPU | 30             | 6.7%            | 35            | 36                  | 27                | 25                      | 3                 | 5                       | 2                      |
| ImageNet16-120 | Raspi4   | 50             | 8.0%            | 53            | 55                  | 44                | 46                      | 6                 | 4                       | 4                      |
| ImageNet16-120 | Eyeriss  | 101            | 17.8%           | 315           | 317                 | 82                | 81                      | 19                | 20                      | 18                     |
