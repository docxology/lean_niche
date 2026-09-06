# src/tests/python/ — Python Test Suite

pytest suite for the `src/python` package (12 test files):
- `test_lean_runner.py`, `test_lean_runner_extended.py` — Lean execution engine
- `test_orchestrator.py`, `test_logging_config.py` — orchestration/logging
- `test_analysis.py`, `test_data_generator.py` — analysis tools
- `test_visualization.py`, `test_cli.py` — visualization and CLI
- `test_lean_verification.py`, `test_proof_artifact_contents.py`,
  `test_proof_artifacts_integration.py` — verification and artifact integration

Run: `pytest src/tests/python/ -v` (config in root `pytest.ini`), or
`python src/tests/run_tests.py --python-only`.
