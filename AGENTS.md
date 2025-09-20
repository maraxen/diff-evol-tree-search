# Diff-Evol-Tree-Search Agents Instructions

These instructions guide LLMs in understanding the codebase, adhering to best practices, and contributing effectively to the **diff-evol-tree-search** project.

---

## 1. Project Overview and Goals

This project explores differentiable algorithms for phylogenetic tree search, leveraging the JAX ecosystem to create models that can be JIT-compiled and optimized via gradient-based methods. The aim is to provide a modern, performant alternative to traditional heuristic search algorithms.

**Primary Goals:**

- **Differentiable Phylogenetics:** Implement and experiment with differentiable versions of classic phylogenetic algorithms (e.g., Sankoff's algorithm for maximum parsimony) in a functional style.
- **JAX Best Practices:** Ensure all components are written using idiomatic JAX, leveraging its functional programming paradigm for clarity and performance.
- **Modular & Extensible:** Maintain a modular structure to facilitate easy testing, debugging, and extension with new algorithms or models.
- **Reproducibility & Clarity:** Ensure that the code is well-documented, tested, and easy for researchers to understand and build upon.

---

## 2. Core Principles & Development Practices

Adherence to these principles is paramount for maintaining code quality and consistency.

### A. JAX Idioms and Functional Programming

- **JAX-Compatible Code:** All new numerical operations should use JAX's functional programming paradigm.
- **Immutability:** Favor immutable data structures. Handle state explicitly by passing and returning updated versions.
- **JIT/Vmap/Scan Compatibility:** Ensure functions are compatible with JAX's `jit`, `vmap`, and `scan` for performance. Use `vmap` for efficient batching.
- **Static Arguments:** Use `static_argnums` for function arguments that are not JAX types and do not change across JAX transformations.
- **Documentation:** Use Google-style docstrings for all functions, including comprehensive type hints and examples.

### B. Code Quality & Linting (Ruff)

- **Linter:** Use Ruff for all linting and formatting tasks.
- **Configuration:** Adhere strictly to the `pyproject.toml` settings:
  - `select = ["ALL"]` (all rules enabled by default)
  - `ignore = ["PD"]` (pandas-specific rules are ignored)
  - `line-length = 100`
  - `indent-width = 2`
  - `fix = true` (Ruff's autofix capabilities should be utilized)
- **Execution:** Run `ruff check . --fix` regularly to apply automatic fixes.
- **Fix Failure Threshold:** If automated `ruff --fix` attempts fail more than 5 times consecutively on the same set of issues, cease further attempts and flag the code for manual review.

### C. Type Checking (Pyright)

- **Evaluator:** Use Pyright for static type checking.
- **Strict Mode:** All code must pass Pyright in strict mode. Ensure type hints are accurate and comprehensive.

### D. Testing

- **Availability:** All new components and features must be accompanied by comprehensive unit and/or integration tests.
- **Framework:** Use `pytest`.
- **Test Location:** Place tests in the `tests/` directory, mirroring the structure of the source code (e.g., a test for `modules/sankoff.py` should be in `tests/test_sankoff.py`).
- **Execution:** Run tests with `python -m pytest tests/`.
- **Test Philosophy:** Tests should cover typical use cases, edge cases, and ensure the correctness of numerical outputs.
- **Test Coverage:** Aim for high test coverage (target: 90%), especially for core algorithmic components. Use `pytest-cov` for reporting.
- **Test Data:** Use fixtures or mock data where necessary to ensure tests are deterministic.

### E. Code Structure & Modularity

- **Modularity:** Maintain the modular structure within the `modules/` directory. Each file should have a clear and distinct responsibility.
- **DRY Principle:** Adhere to the "Don't Repeat Yourself" (DRY) principle. Refactor common logic into reusable functions.
- **Dependencies:** Avoid adding unnecessary external dependencies. Keep `requirements.txt` (or the `pyproject.toml` dependency list) minimal and up-to-date.

---

## 3. Interaction Guidelines

- Prioritize idiomatic JAX solutions.
- Strictly adhere to Ruff linting rules and Pyright strict mode for type checking.
- Always include appropriate tests for new code.
- If automated fixes for linting or type errors fail repeatedly (more than 5 times for the same issue), report to the user for manual review.
- Refer to the `readme.md` for the project's scientific context and general overview.

---

## 4. Example Function Documentation

```python
"""This is a template for a Google-style docstring.

Args:
  param1 (int): The first parameter.
  param2 (str): The second parameter.

Returns:
  bool: The return value. True for success, False otherwise.

Raises:
  IOError: An error occurred accessing the file.

Example:
  >>> example_function(1, "hello")
  True
"""
```
