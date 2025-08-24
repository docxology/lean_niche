"""Compatibility package shim so imports like `python.core.*` resolve to `src/python/core`.

This module adjusts the package search path at runtime so the real implementation
under `src/python` is used. This keeps existing import statements stable while
making tests and runtime imports work from the repository root.
"""
from __future__ import annotations
import pathlib
import os

# Prepend the actual src/python directory to this package's __path__ so that
# `import python.core.lean_runner` will resolve to `src/python/core/lean_runner.py`.
_pkg_dir = pathlib.Path(__file__).parent
# repo root is parent of the `python` package directory
_repo_root = _pkg_dir.parent
_src_python = str((_repo_root / 'src' / 'python').resolve())
if _src_python not in __path__:
    __path__.insert(0, _src_python)


