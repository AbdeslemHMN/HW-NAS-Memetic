# Architectural Genotype Uniqueness Summary
_Generated: 2026-05-11 21:01 UTC_

## Overview

This report measures how many distinct architecture strings (genotypes)
appear on the Pareto front of each algorithm, and how many of the
Proposed algorithm's Pareto-optimal architectures were **not** discovered
by any baseline.

**Novelty score** = (unique-to-proposed / proposed front size) × 100%

## Summary Table

| Dataset        | Hardware | Proposed Front | Global Novelty% | NSGA-II Front | Random Search Front | Shared vs NSGA-II | Shared vs Random Search | Unique vs NSGA-II | Unique vs Random Search | Fully Unique (all BLs) |
| -------------- | -------- | -------------- | --------------- | ------------- | ------------------- | ----------------- | ----------------------- | ----------------- | ----------------------- | ---------------------- |
| cifar10        | Edge GPU | 28             | 0.0%            | 28            | 28                  | 28                | 26                      | 0                 | 2                       | 0                      |
| cifar10        | Raspi4   | 45             | 11.1%           | 43            | 42                  | 40                | 38                      | 5                 | 7                       | 5                      |
| cifar10        | Eyeriss  | 365            | 0.0%            | 366           | 360                 | 364               | 358                     | 1                 | 7                       | 0                      |
| cifar100       | Edge GPU | 24             | 4.2%            | 25            | 27                  | 23                | 21                      | 1                 | 3                       | 1                      |
| cifar100       | Raspi4   | 39             | 0.0%            | 39            | 39                  | 39                | 39                      | 0                 | 0                       | 0                      |
| cifar100       | Eyeriss  | 298            | 0.0%            | 299           | 296                 | 294               | 291                     | 4                 | 7                       | 0                      |
| ImageNet16-120 | Edge GPU | 36             | 0.0%            | 35            | 36                  | 35                | 33                      | 1                 | 3                       | 0                      |
| ImageNet16-120 | Raspi4   | 52             | 1.9%            | 53            | 55                  | 46                | 51                      | 6                 | 1                       | 1                      |
| ImageNet16-120 | Eyeriss  | 279            | 4.7%            | 315           | 317                 | 257               | 259                     | 22                | 20                      | 13                     |
