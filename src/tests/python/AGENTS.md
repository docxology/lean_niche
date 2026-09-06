# Python Tests

**Layout**: Comprehensive pytest suite with test_*.py files
**Analysis tests**: test_analysis.py, test_data_generator.py
**Integration**: test_lean_runner.py, test_lean_runner_extended.py, test_lean_verification.py
**Proof testing**: test_proof_artifact_contents.py, test_proof_artifacts_integration.py
**Infrastructure**: test_cli.py, test_logging_config.py, test_orchestrator.py, test_visualization.py
**Package structure**: __init__.py for test imports

Comprehensive Python test suite for all system components.
See README.md for pytest configuration and coverage targets.
Verify: `python -m pytest src/tests/python/` for complete Python test run.
