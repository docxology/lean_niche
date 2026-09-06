# Python Source

**Layout**: analysis/ cli.py control/ core/ dynamical/ integration/ statistical/ utils/ visualization/
**Domain modules**: analysis/, control/, dynamical/, integration/, statistical/
**Core infrastructure**: core/ contains orchestrator and Lean runner
**Utilities**: utils/, visualization/ for support functions
**CLI**: cli.py, visualization.py for user interfaces
**Orchestration**: lean_runner.py, orchestrator_base.py
**Package structure**: __init__.py files throughout

Python orchestration layer for Lean mathematical environment.
See README.md for Python-Lean integration patterns.
Verify: `python -m pytest src/tests/python/` for Python test suite.
