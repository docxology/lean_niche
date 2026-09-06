"""Tests for setup verification logic (src/python/utils/setup_verifier.py)."""

import pytest

from src.python.utils.setup_verifier import SetupVerifier


@pytest.fixture
def verifier(tmp_path):
    """SetupVerifier rooted at an empty temp project."""
    return SetupVerifier(tmp_path)


class TestFileChecks:
    """Single-path existence/executability checks"""

    def test_missing_file_reported(self, verifier):
        assert verifier.verify_file_exists("lakefile.toml", "config file lakefile.toml") is False
        assert verifier.issues == ["Missing config file lakefile.toml: lakefile.toml"]

    def test_present_file_passes(self, verifier, tmp_path):
        (tmp_path / "lakefile.toml").write_text("[project]\n")
        assert verifier.verify_file_exists("lakefile.toml", "config file lakefile.toml") is True
        assert verifier.issues == []

    def test_missing_directory_reported(self, verifier):
        assert verifier.verify_directory_exists("src/lean", "directory src/lean") is False
        assert verifier.issues == ["Missing directory src/lean: src/lean"]

    def test_file_is_not_directory(self, verifier, tmp_path):
        (tmp_path / "not_a_dir").write_text("")
        assert verifier.verify_directory_exists("not_a_dir", "directory") is False

    def test_executable_flag(self, verifier, tmp_path):
        script = tmp_path / "run.sh"
        script.write_text("#!/bin/sh\n")
        script.chmod(0o755)
        assert verifier.verify_executable("run.sh", "script") is True

        script.chmod(0o644)
        assert verifier.verify_executable("run.sh", "script") is False
        assert verifier.issues == ["Not executable script: run.sh"]


class TestSourceChecks:
    """Directory inventory checks"""

    def test_python_files_complete(self, verifier, tmp_path):
        pydir = tmp_path / "src" / "python"
        pydir.mkdir(parents=True)
        for name in ("__init__.py", "cli.py", "visualization.py"):
            (pydir / name).write_text("")
        assert verifier.verify_python_files() is True

    def test_python_files_missing_entry_reported(self, verifier, tmp_path):
        pydir = tmp_path / "src" / "python"
        pydir.mkdir(parents=True)
        assert verifier.verify_python_files() is False
        assert "Missing Python file: cli.py" in verifier.issues

    def test_lean_files_missing_dir(self, verifier):
        assert verifier.verify_lean_files() is False
        assert verifier.issues == ["Missing src/lean directory"]

    def test_config_files_tolerates_missing_toml_lib(self, verifier, tmp_path):
        # toml is an optional dependency; the check must not raise either way
        (tmp_path / "lakefile.toml").write_text("[project]\n")
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "lean-toolchain").write_text("v4.22.0\n")
        (tmp_path / ".env.example").write_text("")
        assert verifier.verify_config_files() is True
        assert verifier.issues == []
