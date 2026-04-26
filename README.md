# HW-NAS-Memetic: A Multi-Level Hybrid Metaheuristic for Edge AI

[Badges here: e.g., Python Version, License, Build Status]

## 1. Abstract & Motivation
Deploying Deep Learning models on Edge AI devices (like smartphones, IoT sensors, and UAVs) requires strictly balancing two conflicting objectives: maximizing **Accuracy** while minimizing hardware constraints like **Latency**. Hardware-Aware Neural Architecture Search (HW-NAS) automates this process. 

Currently, the State-of-the-Art in multi-objective HW-NAS is dominated by purely Evolutionary Algorithms (e.g., NSGA-II). However, in discrete, cell-based search spaces like NAS-Bench-201, neural architectures are represented as Directed Acyclic Graphs (DAGs) exhibiting extreme **epistasis**—meaning the operations within a network are deeply interdependent. Traditional evolutionary crossover violently breaks these dependencies by slicing and combining disparate graphs, frequently resulting in catastrophic performance collapse.

**HW-NAS-Memetic** proposes a novel solution: a Multi-Level Hybrid Metaheuristic. Instead of relying solely on destructive global crossover, this framework integrates the global scalarization of **MOEA/D** with the topology-preserving local exploitation of a custom **Discrete Probabilistic PSO** and **Simulated Annealing**. This ensures that architectures can safely optimize their hardware latency bottlenecks edge-by-edge without destroying their functional graph routing.

## 2. Why "Memetic"? (The Philosophy Behind the Name)
*n optimization literature, there is a strict mathematical distinction between *Genetic* and *Memetic* algorithms, rooted in evolutionary biology:

* 🧬 **Genetic Algorithms (e.g., NSGA-II):** Operate strictly on Darwinian evolution. An architecture is "born", evaluated, and it either breeds or dies. It undergoes **no lifetime learning**.
* 🧠 **Memetic Algorithms (Cultural Evolution):** Combine population-based global search with *lifelong learning*. An architecture is generated, but before its final evaluation, it is given the tools to explore its local neighborhood and actively fix its own flaws.

**Why is this critical for NAS?**
If we strictly use Genetics (NSGA-II crossover), combining the front half of a deep network with the back half of a wide network destroys the co-adapted data flow. 

By utilizing a **Memetic** approach, our algorithm generates a baseline architecture globally (via Discrete GA), but then grants it a "lifetime" to learn. Using our custom **Discrete PSO**, the architecture consults its personal memory (`pbest`) and the swarm's memory (`gbest`), actively tweaking single edges (e.g., swapping a 3x3 Conv for a Skip Connection) to shave off milliseconds of latency before committing to the Pareto Front. 

This memetic refinement acts as a highly effective, topology-safe local search that pure evolutionary algorithms lack.*

## 3. Key Innovations & Algorithm Architecture
*(We will list the core components: MOEA/D scalarization, the Dual-Generation Engine (Discrete GA + Discrete Probabilistic PSO), Metropolis Acceptance, and A Posteriori Pareto Filtering.)*

## 4. Search Space & Problem Encoding
*(We will define the HW-NAS-Bench environment: the 4-node, 6-edge DAG, the 5 operations, and the $O(1)$ dictionary lookup for Accuracy and Latency.)*

## 5. Repository Structure
*(We will paste the folder tree we designed earlier so researchers can easily navigate the codebase.)*

## 6. Installation & Environment Setup
*(We will provide the `pip install` commands and instructions on where to download and place the `HW-NAS-Bench-v1_0.pickle` dataset.)*

## 7. Quickstart & Usage
*(We will provide the exact command-line examples to run the search, e.g., `python scripts/run_search.py --budget 2000`.)*

## 8. Baselines & Evaluation Metrics
*(We will explain how the repository compares our algorithm against Random Search and `pymoo` NSGA-II using Hypervolume (HV) and Inverted Generational Distance (IGD).)*

## 9. Implemented Upgrades (Advanced Features)
*(We will list the 4 critical improvements we discussed: Self-Adaptive Parameters, Multi-Armed Bandit Tool Selection, Dynamic Weight Adaptation, and Block-wise PSO.)*

## 10. Citation & Acknowledgements
*(We will provide a BibTeX format for referencing your work and acknowledge NAS-Bench-201 and HW-NAS-Bench.)*