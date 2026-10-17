from src.python.core.lean_runner import LeanRunner


def test_real_compiler_accepts_exact_proof():
    result = LeanRunner().run_lean_code("theorem exactWitness : (1 : Nat) + 1 = 2 := rfl")
    assert result["success"], result
    assert result["result"]["verification_status"]["compilation_successful"]


def test_real_compiler_error_cannot_be_reported_as_success():
    result = LeanRunner().run_lean_code("theorem falseWitness : False := by trivial")
    assert not result["success"]
    assert result["result"]["theorems_proven"] == []


def test_real_compiler_admission_is_not_verified():
    result = LeanRunner().run_lean_code("theorem admittedWitness : False := by sorry")
    assert not result["success"]
    assert result["result"]["theorems_proven"] == []


def test_unverified_artifact_cannot_inflate_proof_counts(tmp_path):
    import hashlib
    import json
    from pathlib import Path

    runner = LeanRunner()
    project = Path(__file__).resolve().parents[3] / 'lean'
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in project.rglob('*.lean')}
    (tmp_path / 'unrelated.lean').write_text('theorem fakeArtifact : False := by sorry\n')
    result = runner.run_lean_code('theorem validWitness : (2 : Nat) = 2 := rfl')
    saved = runner.save_comprehensive_proof_outcomes(result, tmp_path)
    runner._consolidate_lean_artifacts(saved, tmp_path, 'control')
    proofs = json.loads(saved['theorems'].read_text())
    assert all(p['name'] != 'fakeArtifact' for p in proofs['theorems_proven'])
    inventory = json.loads(saved['source_inventory'].read_text())
    assert inventory['artifacts'][0]['verification_status'] == 'unverified_source'
    assert saved['complete_summary'].exists()
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in project.rglob('*.lean')}
    assert after == before


def test_custom_axiom_cannot_certify_a_theorem():
    result = LeanRunner().run_lean_code('axiom unsupported : False\ntheorem unsupportedWitness : False := unsupported')
    assert not result['success']
    assert result['result']['theorems_proven'] == []


def test_comment_declarations_are_not_proof_evidence():
    result = LeanRunner().run_lean_code('-- theorem imaginary : False := by sorry\nnamespace Exact\ntheorem realWitness : (3 : Nat) = 3 := rfl\nend Exact')
    assert result['success'], result
    assert [d['name'] for d in result['result']['theorems_proven']] == ['Exact.realWitness']


def test_algorithm_tests_include_algorithm_and_compare_expected_values():
    runner = LeanRunner()
    code = 'def double (n : Nat) : Nat := n + n'
    good = runner.run_algorithm_verification(code, [{'expression': 'double 3', 'expected': 6}])
    assert good['success'], good
    bad = runner.run_algorithm_verification(code, [{'expression': 'double 3', 'expected': 7}])
    assert not bad['success'], bad
    assert bad['performance_metrics']['failed_tests'] == 1
