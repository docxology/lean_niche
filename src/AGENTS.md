# Source Code

**Layout**: latex/ lean/ python/ src/ tests/
**Multi-language**: Lean mathematical proofs, Python orchestration, LaTeX output
**Main sources**: lean/ contains LeanNiche library, python/ contains utilities
**Nested src**: src/ subdirectory contains additional Lean code
**Testing**: tests/ contains comprehensive test suites
**Package init**: __init__.py for Python package structure

See README.md for source organization and build boundaries.
Verify: `lake build` for Lean, `python -m pytest src/tests/python/` for Python.
