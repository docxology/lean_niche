import shutil
from pathlib import Path
import pytest

from src.python.core.lean_runner import LeanRunner


def _require_lean():
    if not shutil.which('lean'):
        pytest.skip("Real Lean not found in PATH; skipping integration test")


def test_simple_lean_proof_and_artifacts(tmp_path):
    """Run a trivial Lean file and assert proof artifacts are created."""
    _require_lean()

    lr = LeanRunner()

    lean_code = """namespace TestIntegration
def answer : Nat := 42
theorem answer_eq : answer = 42 := by rfl
end TestIntegration
"""

    res = lr.run_lean_code(lean_code, imports=[])
    assert res.get('success', False), f"Lean execution failed: {res.get('stderr', '')}"

    saved = lr.generate_proof_output(res, tmp_path, prefix='integration_test')
    # Ensure expected artifact keys exist and files exist
    assert 'summary_json' in saved and Path(saved['summary_json']).exists()
    assert 'complete_summary' in saved and Path(saved['complete_summary']).exists()
    # Theorems file should exist
    assert 'theorems' in saved and Path(saved['theorems']).exists()


def test_statistical_orchestrator_generates_proofs(tmp_path):
    _require_lean()

    # Import orchestrator lazily to avoid side effects when Lean missing
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3]))
    from examples.statistical_analysis_example import StatisticalAnalysisOrchestrator

    outdir = tmp_path / 'statistical_out'
    orch = StatisticalAnalysisOrchestrator(output_dir=str(outdir))

    results = orch.run_comprehensive_orchestration()

    proofs_dir = outdir / 'proofs'
    assert proofs_dir.exists(), f"Proofs directory not found: {proofs_dir}"
    # index.json or at least a complete_summary should exist
    assert any(proofs_dir.glob('**/*complete_summary*.json'))


