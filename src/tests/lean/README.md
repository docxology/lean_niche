# src/tests/lean/ — Lean Test Modules

Lean-side tests, compiled with the library:
- `test_basic.lean`, `test_advanced.lean`, `test_comprehensive.lean` — theorem tests across modules
- `test_proof_export.lean` — tests for the ProofExport (Lean side)
- `TestSuite.lean` — unified test suite (root of the `test_suite` lake executable)
- `__init__.lean`

Run via `lake build && lake exe test_suite` (per `lakefile.toml`), or
`python src/tests/run_tests.py --lean-only`.
