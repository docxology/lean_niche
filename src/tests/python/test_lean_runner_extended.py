"""Extended tests for LeanRunner export and proof output helpers"""

from pathlib import Path
import json

from src.python.core.lean_runner import LeanRunner


def test_export_lean_code(tmp_path):
    lr = LeanRunner()
    code = "theorem trivial : True := by trivial"
    out = tmp_path / 'out.lean'
    p = lr.export_lean_code(code, out)
    assert p.exists()
    content = out.read_text(encoding='utf-8')
    assert 'theorem trivial' in content


def test_generate_proof_output(tmp_path):
    lr = LeanRunner()
    results = {'success': True, 'result': {'theorems_proven': []}, 'stdout': '', 'stderr': ''}
    saved = lr.generate_proof_output(results, tmp_path, prefix='unit_test')
    # check that summary JSON is present
    assert 'summary_json' in saved
    assert Path(saved['summary_json']).exists()
    # check that complete summary file exists
    assert 'complete_summary' in saved


def test_suggest_atp_integration():
    lr = LeanRunner()
    guidance = lr.suggest_atp_integration()
    assert 'ATP' in guidance or 'ATP/SMT' in guidance or 'ATP' in guidance


