# AGENTS: `scripts/` — Thin Orchestrator Contract

## Contract

- Scripts in `scripts/` are thin orchestrators ONLY: argparse/CLI handling,
  path bootstrap, logging, and a single delegated call into a `src/python/`
  entrypoint (or the Lake/Lean toolchain for shell scripts).
- Business, data, analysis, and verification logic lives in `src/python/`,
  importable and covered by tests in `src/tests/python/`. Never add logic to
  a script: move it into the delegated module and call it from the script.
- New scripts MUST be added to the inventory in this file and in README.md.

## Inventory

| Script | Delegates to | Notes |
|--------|--------------|-------|
| `build.sh` | `lake clean`/`lake build`/`lake exe lean_niche`, `lean` | Multi-target (`clean\|build\|test\|docs\|analyze\|perf\|verify\|lean\|all`); shell-level orchestration only |
| `test.sh` | `lake build`, `lean` | Lean-only suite; Python tests run via pytest, not this script |
| `analyze.sh` | inline `find`/`wc`/`grep`/`lean` reporting | Metrics/reporting; no src module yet |
| `docs.sh` | `python3` import check, `lean`/`grep` extraction | Writes `docs/code_metrics.md` and usage examples |
| `verify_setup.py` | `src/python/utils/setup_verifier.py` (`SetupVerifier`) | Sanctioned role: environment check; logic moved out of the script 2026-09 |
| `verify_lean_proofs.py` | `lean` (logic currently inline) | Legacy deviation: verification logic not yet migrated to `src/` |

## Gotchas

- `verify_setup.py` bootstraps `sys.path` to the repo root before importing
  `src.python.utils.setup_verifier`; run it as
  `uv run python scripts/verify_setup.py`.
- `toml` is NOT a project dependency: `SetupVerifier` treats it as optional
  and skips the `lakefile.toml` syntax check when absent.
- `uv run pytest -q` fails at collection (`No module named 'src.python'`);
  the documented test command is `uv run python -m pytest src/tests/python/`
  (`python -m` puts the repo root on `sys.path`).
- `make test` runs the Lean-only `test.sh`; it does not run the Python suite.
- `examples/statistical_analysis_example.py` imports
  `scripts.verify_lean_proofs` directly (legacy private script import).
- Keep the directory free of `__pycache__/` and `.DS_Store` (gitignored).
