# Manuscript status — LeanNiche

**Repo type:** Tool / research-environment library. A Lean 4 deep-research
environment (proof logging, verification, statistics/dynamical-systems
utilities) with a Python companion package (`pyproject.toml`: name
`lean-niche`, "Python utilities for LeanNiche mathematical research
environment") and a Makefile-driven workflow (`make setup|build|test|docs|viz`).

**Evidence checked:** repo root listing (`README.md`, `pyproject.toml`,
`Makefile`, `lakefile.toml`, `lean-toolchain`, `src/`, `tests/`,
`examples/`, `docs/`), `docs/` index and topic files, `outputs/` (generated
execution reports). No `manuscript/` or `docs/manuscript/` directory existed
before this file; the repo ships research documentation and verification
reports but no publication-track paper source.

**Why no publication-target manuscript applies today:** the repository is
infrastructure for producing verified mathematics, not a narrative research
output; its deliverables are code, proofs, and generated reports under
`outputs/`.

**What would trigger creating one:** a methods/publication paper describing
the LeanNiche environment itself (e.g. a "tool paper" covering the
verification workflow, logging/verification guarantees, and the statistics
and dynamical-systems libraries), or a research paper whose results are
produced with LeanNiche. At that point, add a full `manuscript/` tree at the
repo top level (config.yaml, section files 00–99, references.bib) following
the docxology/template standard.
