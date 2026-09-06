# LeanNiche Research Environment

**Layout**: .github/ docs/ examples/ outputs/ scripts/ src/
**Build system**: Lake (lakefile.toml), Python (pyproject.toml), Lean 4 toolchain v4.22.0
**Generated artifacts**: lake-manifest.json, outputs/*/, proof_*.lean files
**Key scripts**: scripts/build.sh, scripts/verify_lean_proofs.py, main.py
**Mixed language boundaries**: Lean src/lean/, Python src/python/, LaTeX src/latex/
**Testing**: src/tests/ with Lean TestSuite.lean and Python pytest suite
**Docker**: Available via Makefile targets

Verify: `lake build && python -m pytest src/tests/python/`
See README.md for full setup and architecture overview.
