# Project Plan: Differentiable SVM Tree

**Last Updated**: 2025-10-02

This document tracks the development of the Differentiable SVM Tree. Our guiding principles are **modularity, testability, and reproducibility**.

---

## Phase 0: Project Setup & CI/CD Hardening

*Status: All tasks in this phase must be completed before proceeding.*

| Task ID | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **0.1** | Establish Project Scaffolding (`/scripts`, `/svm_tree`) | 📝 **To Do** | |
| **0.2** | Configure Unified Training Script & Dataclass Configs | 📝 **To Do** | `scripts/train.py`, `src/trex/svm_tree/configs.py` |
| **0.3** | Implement CI/CD with `uv` | 📝 **To Do** | GH Actions: `uv venv`, `uv pip install`, then run `ruff`, `pyright`, `pytest`. |

---

## Phase 1: Core Component Development (Parallel)

**Architectural Guideline**: All stateful components **must be `equinox.Module` subclasses** to cleanly separate parameters from code logic.

*Definition of Done: >90% test coverage, full documentation, and passing CI checks for all new modules.*

| Task ID | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **1.A.1** | Implement `svm.py` as an `equinox.Module` | 📝 **To Do** | Location: `.../components/svm.py` |
| **1.A.2** | Implement SVM Unit Tests | 📝 **To Do** | |
| **1.B.1** | Implement `data_utils.py` for MNIST | 📝 **To Do** | |
| **1.B.2** | Define Dataclasses in `configs.py` for MNIST | 📝 **To Do** | |
| **1.B.3**| *(Ongoing)* Source & Prepare Plant Dataset | 📝 **To Do** | Research task. |

---

## Phase 2: System Integration & Validation

*Goal: Prove the end-to-end viability of the core architecture with a fixed tree.*

| Task ID | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **2.1** | Implement `BaseTreeModel` with Fixed Topology | 📝 **To Do** | Depends on Phase 1 |
| **2.2** | Integrate and run MNIST experiment via `scripts/train.py` | 📝 **To Do** | Depends on 2.1 |

---

## Phase 3: Introducing Dynamic Topologies

*Goal: Replace the fixed tree with a learnable, differentiable topology.*

| Task ID | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **3.1** | Create `DifferentiableTopology` Component | 📝 **To Do** | Depends on Phase 2 |
| **3.2** | Create `LearnableTreeModel` via Composition | 📝 **To Do** | Depends on 3.1 |
| **3.3** | Run Learnable Tree Experiment on MNIST | 📝 **To Do** | Depends on 3.2 |

---

## Phase 4: Multi-Task Generalization & Analysis

*Goal: Implement and test the final multi-task architecture.*

| Task ID | Description | Status | Notes |
| :--- | :--- | :--- | :--- |
| **4.1** | Implement `TaskConditionalFFN` Component | 📝 **To Do** | Depends on Phase 3|
| **4.2** | Implement `MultiTaskModel` via Composition | 📝 **To Do** | Depends on 4.1 |
| **4.3** | Integrate Plant Data & Configs | 📝 **To Do** | Depends on 1.B.3, 4.2 |
| **4.4** | Implement Visualization Utilities | 📝 **To Do** | `viz_utils.py`, tree plotting, gate activations |
| **4.5** | Run Final Experiment & Analyze | 📝 **To Do** | |