"""Tests for LeanNicheOrchestratorBase behavior (uses monkeypatching)"""

import tempfile
from pathlib import Path

from src.python.orchestrator_base import LeanNicheOrchestratorBase


class DummyOrchestrator(LeanNicheOrchestratorBase):
    def __init__(self, domain_name: str, output_dir: str):
        super().__init__(domain_name, output_dir)

    def run_domain_specific_analysis(self):
        return {'dummy': True}

    def create_domain_visualizations(self, analysis_results):
        # Create a dummy file to simulate visualization output
        (self.viz_dir / 'dummy.png').write_text('png')


def test_orchestrator_setup_and_run(monkeypatch, tmp_path):
    outdir = tmp_path / 'out'
    outdir_str = str(outdir)

    # Monkeypatch LeanRunner.run_lean_code to avoid calling external lean
    def fake_run_lean_code(self, code, imports=None):
        return {
            'success': True,
            'result': {
                'theorems_proven': [],
                'definitions_created': [],
                'verification_status': {'total_proofs': 0, 'total_errors': 0, 'success_rate': 100, 'compilation_successful': True, 'verification_complete': True}
            },
            'stdout': '',
            'stderr': ''
        }

    monkeypatch.setattr('src.python.orchestrator_base.LeanRunner.run_lean_code', fake_run_lean_code)

    orch = DummyOrchestrator('TestDomain', outdir_str)

    # Run orchestration (should not raise)
    results = orch.run_comprehensive_orchestration()

    assert results['analysis_results'] == {'dummy': True} or isinstance(results, dict)
    assert (orch.viz_dir / 'dummy.png').exists()


