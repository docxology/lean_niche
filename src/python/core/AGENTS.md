# Python Core Infrastructure

**Layout**: lean_runner.py, lean_source.py, logging_config.py, orchestrator_base.py
**Lean integration**: lean_runner.py for Python-Lean communication
**Logging**: logging_config.py for system-wide logging configuration
**Orchestration**: orchestrator_base.py for workflow coordination
**Package structure**: __init__.py for module imports

Core Python infrastructure for Lean integration.
See README.md for orchestrator patterns and Lean runner usage.
Verify: `python -c "import src.python.core; print('OK')"` for core import.
