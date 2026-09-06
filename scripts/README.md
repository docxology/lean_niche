# scripts/ — Development Scripts

Thin orchestrators for day-to-day development: argument parsing, path
bootstrap, logging, and a single delegated call into `src/python/` or the
Lake/Lean toolchain. Testable logic lives in `src/` (contract in AGENTS.md).

## Inventory

| Script | Purpose | Delegates to | Run command |
|--------|---------|--------------|-------------|
| `build.sh` | Lake build lifecycle + per-file Lean compilation checks (targets: `clean`, `build`, `test`, `docs`, `analyze`, `perf`, `verify`, `lean`, `all`) | `lake clean` / `lake build` / `lake exe lean_niche`, `lean` | `./scripts/build.sh [target]` |
| `test.sh` | Lean module/file compilation and import-resolution tests with pass/fail summary | `lake build`, `lean` (per file) | `./scripts/test.sh` |
| `analyze.sh` | Code metrics: structure, theorems, dependencies, complexity, documentation coverage, recommendations | inline `find`/`wc`/`grep`/`lean` reporting | `./scripts/analyze.sh [structure\|theorems\|dependencies\|complexity\|documentation\|recommendations\|lean\|all]` |
| `docs.sh` | Documentation generation: CLI import check, code metrics (`docs/code_metrics.md`), usage examples | `python3` import check, `lean`/`grep` doc extraction | `./scripts/docs.sh` |
| `verify_setup.py` | Environment/setup verification: structure, config files, Lean compilation + imports, Python/LaTeX sources, scripts | `src/python/utils/setup_verifier.py` (`SetupVerifier`) | `uv run python scripts/verify_setup.py` |
| `verify_lean_proofs.py` | Per-file Lean proof verification → JSON report | `lean` compiler (verification logic currently inline — see AGENTS.md) | `python scripts/verify_lean_proofs.py <lean_file> <output_prefix>` |

Entry point: `make setup` → `./setup.sh`. Canonical invocations live in the
root `Makefile`. Python tests: `uv run python -m pytest src/tests/python/`.
