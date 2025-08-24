"""Tests for LeanRunner parsing helpers (no Lean binary required)"""

from src.python.core.lean_runner import LeanRunner


def test_parse_error_line():
    lr = LeanRunner()
    line = "error: Syntax error at file.lean:12:5"
    parsed = lr._parse_error_line(line)
    assert parsed['type'] in ['syntax', 'unknown']
    assert parsed['severity'] == 'error'


def test_parse_warning_line():
    lr = LeanRunner()
    line = "warning: unused variable at file.lean:5:3"
    parsed = lr._parse_warning_line(line)
    assert parsed['type'] == 'unused_variable' or parsed['type'] == 'unknown'
    assert parsed['severity'] == 'warning'


