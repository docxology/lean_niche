import shutil
import sys
from pathlib import Path
import importlib
import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parents[3] / 'src'))


def _require_lean():
    if not shutil.which('lean'):
        pytest.skip("Real Lean not found in PATH; skipping integration test")


def _read_all_json_strings(dir_path: Path):
    texts = []
    for p in dir_path.glob('**/*.json'):
        try:
            texts.append(p.read_text(encoding='utf-8'))
        except Exception:
            continue
    return '\n'.join(texts)


def test_control_theory_proof_artifacts(tmp_path):
    _require_lean()
    # dynamic import to avoid static resolution warnings
    mod = importlib.import_module('examples.control_theory_example')
    ControlTheoryOrchestrator = getattr(mod, 'ControlTheoryOrchestrator')

    out = tmp_path / 'control'
    orch = ControlTheoryOrchestrator(output_dir=str(out))
    orch.run_comprehensive_orchestration()

    proofs_dir = Path(orch.proofs_dir)
    assert proofs_dir.exists()

    content = _read_all_json_strings(proofs_dir)
    # check for trivial artifact theorem name
    assert 'num_pid_controllers_eq' in content or 'num_pid_controllers' in content


def test_dynamical_systems_proof_artifacts(tmp_path):
    _require_lean()
    mod = importlib.import_module('examples.dynamical_systems_example')
    DynamicalSystemsOrchestrator = getattr(mod, 'DynamicalSystemsOrchestrator')

    out = tmp_path / 'dynamical'
    orch = DynamicalSystemsOrchestrator(output_dir=str(out))
    orch.run_comprehensive_orchestration()

    proofs_dir = Path(orch.proofs_dir)
    assert proofs_dir.exists()

    content = _read_all_json_strings(proofs_dir)
    assert 'num_parameters_eq' in content or 'sample_iterations_eq' in content


def test_integration_showcase_proof_artifacts(tmp_path):
    _require_lean()
    mod = importlib.import_module('examples.integration_showcase_example')
    IntegrationShowcaseOrchestrator = getattr(mod, 'IntegrationShowcaseOrchestrator')

    out = tmp_path / 'integration'
    orch = IntegrationShowcaseOrchestrator(output_dir=str(out))
    orch.run_comprehensive_orchestration()

    proofs_dir = Path(orch.proofs_dir)
    assert proofs_dir.exists()

    content = _read_all_json_strings(proofs_dir)
    assert 'num_sensors_eq' in content or 'sim_duration_eq' in content


def test_statistical_proof_artifacts(tmp_path):
    _require_lean()
    mod = importlib.import_module('examples.statistical_analysis_example')
    StatisticalAnalysisOrchestrator = getattr(mod, 'StatisticalAnalysisOrchestrator')

    out = tmp_path / 'stat'
    orch = StatisticalAnalysisOrchestrator(output_dir=str(out))
    orch.run_comprehensive_orchestration()

    proofs_dir = Path(orch.proofs_dir)
    assert proofs_dir.exists()

    content = _read_all_json_strings(proofs_dir)
    # we expect some theorems or definitions saved
    assert 'theorems_proven' in content or 'definitions_created' in content


