# LeanNiche Research Environment

**Layout**: .github/ docs/ examples/ outputs/ scripts/ src/
**Build system**: Lake (lakefile.toml), Python (pyproject.toml), Lean 4 toolchain v4.35.0
**Generated artifacts**: lake-manifest.json, outputs/*/
**Key scripts**: scripts/build.sh, scripts/verify_lean_proofs.py, main.py
**Mixed language boundaries**: Lean src/lean/, Python src/python/, LaTeX src/latex/
**Testing**: src/tests/ with Lean TestSuite.lean and Python pytest suite
**Docker**: Available via Makefile targets

Verify: `lake build && python -m pytest src/tests/python/`
See [the documentation index](docs/README.md) for setup and architecture.

The default build includes the public library, CLI and executable regression controls.
Float computation APIs remain unchanged; exact rational proof models and the
proof-API correction are documented in [the exact model guide](docs/exact-models.md).
