# LeanNiche Research Environment

**Layout**: .github/ docs/ examples/ outputs/ scripts/ src/
**Build system**: Lake (lakefile.toml), Python (pyproject.toml), Lean 4 toolchain v4.35.0
**Generated artifacts**: lake-manifest.json, outputs/*/
**Key scripts**: scripts/build.sh, scripts/verify_lean_proofs.py, main.py
**Mixed language boundaries**: Lean src/lean/, Python src/python/, LaTeX src/latex/
**Testing**: src/tests/ with Lean TestSuite.lean and Python pytest suite
**Docker**: Available via Makefile targets

Run verification from the repository root after preparing the Lean compiler
from [lean-toolchain](lean-toolchain) and the locked Python 3.13 environment:

```sh
uv sync --locked --python 3.13
lake --wfail build
lake exe test_suite
uv run --no-sync --python 3.13 python -m pytest src/tests/python/
```

The [deployment guide](docs/deployment.md) covers compiler setup and the
separate compiler-integration battery. The [documentation index](docs/README.md)
links the architecture and manuscript build.

The default build includes the public library, CLI and executable regression controls.
Float computation APIs remain unchanged; exact rational proof models and the
proof-API correction are documented in [the exact model guide](docs/exact-models.md).
