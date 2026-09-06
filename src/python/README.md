# src/python/ — Python Companion Package

The `lean-niche` Python package (`pyproject.toml`): utilities that drive Lean
execution, analyze results, and visualize output.

## Contents
- `cli.py` — command-line interface (entry point)
- `lean_runner.py` — Lean code execution engine with proof-outcome extraction
- `orchestrator_base.py` — base class for example orchestrators
- `visualization.py` — top-level visualization helpers
- `core/` — lean_runner, orchestrator base, logging config
- `analysis/` — mathematical analysis, comprehensive analysis, data generation
- `statistical/`, `dynamical/`, `control/` — per-domain analysis modules
- `integration/` — cross-domain integration analysis
- `visualization/` — plotting tools
- `utils/` — CLI utilities

Install with `pip install -e .`; tests in `src/tests/python/` (pytest).
