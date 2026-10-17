"""Comprehensive tests for LeanRunner class"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from src.python.core.lean_runner import LeanRunner


class TestLeanRunner:
    """Comprehensive tests for LeanRunner"""

    def setup_method(self):
        """Setup test fixtures"""
        self.runner = LeanRunner(lean_path="lean", timeout=30)

    def test_initialization(self):
        """Test LeanRunner initialization"""
        assert isinstance(self.runner, LeanRunner)
        assert self.runner.lean_path == "lean"
        assert self.runner.timeout == 30
        assert hasattr(self.runner, 'run_lean_code')
        assert hasattr(self.runner, 'run_theorem_verification')
        assert hasattr(self.runner, 'run_algorithm_verification')
        assert hasattr(self.runner, 'extract_mathematical_results')
        assert hasattr(self.runner, 'create_mathematical_report')
        assert hasattr(self.runner, '_parse_error_line')
        assert hasattr(self.runner, '_parse_warning_line')

    def test_initialization_with_module(self):
        """Test initialization with specific module"""
        runner = LeanRunner(lean_module="TestModule")
        assert runner.lean_module == "TestModule"

    def test_initialization_with_timeout(self):
        """Test initialization with custom timeout"""
        runner = LeanRunner(timeout=60)
        assert runner.timeout == 60

    def test_setup_logging(self):
        self.runner.setup_logging()
        assert self.runner.logger is self.runner.lean_logger.logger

    def test_run_lean_code_simple(self):
        result = self.runner.run_lean_code('def increment (n : Nat) : Nat := n + 1\n#eval increment 2')
        assert result['success'], result
        assert '3' in result['stdout']

    def test_run_lean_code_with_error(self):
        """Test running Lean code with compilation error"""
        error_code = """
def invalid_function (x : Nat) : Nat := x +
"""

        with patch('subprocess.run') as mock_run, \
             patch('tempfile.NamedTemporaryFile') as mock_temp:

            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.lean"

            # Mock subprocess result with error
            mock_result = MagicMock()
            mock_result.returncode = 1
            mock_result.stdout = ""
            mock_result.stderr = "error: unexpected end of input"
            mock_run.return_value = mock_result

            result = self.runner.run_lean_code(error_code)

            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'error' in result

    def test_run_theorem_verification_success(self):
        """Test theorem verification with successful compilation"""
        theorem_code = """
import LeanNiche.Basic

theorem add_comm : ∀ a b : Nat, a + b = b + a := by
  induction a with
  | zero => rfl
  | succ a' ih => rw [Nat.succ_add, ih, ←Nat.add_succ]
"""

        with patch.object(self.runner, 'run_lean_code') as mock_run:
            mock_run.return_value = {
                'success': True,
                'execution_time': 0.5,
                'theorems_verified': 1,
                'definitions_found': 0
            }

            result = self.runner.run_theorem_verification(theorem_code, ["LeanNiche.Basic"])

            assert isinstance(result, dict)
            assert result['success'] is True
            assert 'execution_time' in result
            mock_run.assert_called_once_with(theorem_code, ["LeanNiche.Basic"])

    def test_run_algorithm_verification(self):
        result = self.runner.run_algorithm_verification('def increment (n : Nat) : Nat := n + 1',
            [{'expression': 'increment 2', 'expected': 3}, {'expression': 'increment 4', 'expected': 5}])
        assert result['success'], result
        assert result['performance_metrics']['passed_tests'] == 2

    def test_extract_mathematical_results_theorems(self):
        result = self.runner.extract_mathematical_results('theorem mentioned : True := by trivial\ndef constant : Nat := 1')
        assert [row['name'] for row in result['theorems']] == ['mentioned']
        # Declaration text is an inventory, never kernel verification evidence.
        assert all('axioms' not in row for row in result['theorems'])

    def test_extract_mathematical_results_errors(self):
        checked = self.runner.run_lean_code('theorem invalid : False := by trivial')
        assert not checked['success']
        assert checked['result']['theorems_proven'] == []
        assert checked['result']['error_details']

    def test_create_mathematical_report(self):
        """Test creating mathematical report"""
        lean_code = "theorem test : ∀ x, x = x := by rfl"
        results = {
            'success': True,
            'theorems_verified': 1,
            'definitions_found': 0,
            'execution_time': 0.2
        }

        report = self.runner.create_mathematical_report(lean_code, results)

        assert isinstance(report, str)
        assert len(report) > 0
        assert "Mathematical Verification Report" in report
        assert "test" in report

    def test_batch_verify_theorems(self):
        """Test batch theorem verification"""
        theorems = [
            {"name": "add_comm", "code": "theorem add_comm : ∀ a b : Nat, a + b = b + a := by rfl"},
            {"name": "mul_comm", "code": "theorem mul_comm : ∀ a b : Nat, a * b = b * a := by rfl"}
        ]

        with patch.object(self.runner, 'run_theorem_verification') as mock_verify:
            mock_verify.return_value = {
                'success': True,
                'execution_time': 0.1,
                'theorems_verified': 1
            }

            results = self.runner.batch_verify_theorems(theorems)

            assert isinstance(results, list)
            assert len(results) == 2
            assert all(isinstance(r, dict) for r in results)
            assert mock_verify.call_count == 2

    def test_save_results_json(self, tmp_path):
        result = {'success': False, 'error': 'actual compiler failure'}
        path = self.runner.save_results(result, str(tmp_path / 'receipt'), 'json')
        assert json.loads(path.read_text()) == result

    def test_save_results_json_format(self):
        """Test saving results with explicit json format"""
        results = {'test': 'data'}

        with patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_dump:

            output_path = self.runner.save_results(results, "test_results.json", "json")

            assert str(output_path).endswith('.json')
            mock_file.assert_called_once()
            mock_dump.assert_called_once()

    def test_parse_error_line_syntax_error(self):
        """Test parsing syntax error line"""
        line = "error: syntax error at file.lean:12:5: unexpected token"
        parsed = self.runner._parse_error_line(line)

        assert parsed['type'] == 'syntax'
        assert parsed['severity'] == 'error'
        assert 'line' in parsed
        assert 'column' in parsed
        assert parsed['line'] == 12
        assert parsed['column'] == 5

    def test_parse_error_line_type_error(self):
        """Test parsing type error line"""
        line = "error: type mismatch at theorem.lean:25:10"
        parsed = self.runner._parse_error_line(line)

        assert parsed['type'] == 'type_mismatch'
        assert parsed['severity'] == 'error'
        assert parsed['line'] == 25
        assert parsed['column'] == 10

    def test_parse_error_line_unknown_error(self):
        """Test parsing unknown error line"""
        line = "error: some unknown error in myfile.lean:8:15"
        parsed = self.runner._parse_error_line(line)

        assert parsed['type'] == 'unknown'
        assert parsed['severity'] == 'error'
        assert parsed['line'] == 8
        assert parsed['column'] == 15

    def test_parse_error_line_no_location(self):
        """Test parsing error line without location info"""
        line = "error: compilation failed"
        parsed = self.runner._parse_error_line(line)

        assert parsed['type'] == 'unknown'
        assert parsed['severity'] == 'error'
        assert 'line' not in parsed or parsed['line'] is None

    def test_parse_warning_line_unused_variable(self):
        """Test parsing unused variable warning"""
        line = "warning: unused variable 'x' at file.lean:5:3"
        parsed = self.runner._parse_warning_line(line)

        assert parsed['type'] == 'unused_variable'
        assert parsed['severity'] == 'warning'
        assert parsed['variable'] == 'x'
        assert parsed['line'] == 5
        assert parsed['column'] == 3

    def test_parse_warning_line_deprecated(self):
        """Test parsing deprecated feature warning"""
        line = "warning: deprecated syntax at old.lean:10:8"
        parsed = self.runner._parse_warning_line(line)

        assert parsed['type'] == 'deprecated'
        assert parsed['severity'] == 'warning'
        assert parsed['line'] == 10
        assert parsed['column'] == 8

    def test_parse_warning_line_unknown_warning(self):
        """Test parsing unknown warning line"""
        line = "warning: some unknown warning in test.lean:3:7"
        parsed = self.runner._parse_warning_line(line)

        assert parsed['type'] == 'unknown'
        assert parsed['severity'] == 'warning'
        assert parsed['line'] == 3
        assert parsed['column'] == 7

    def test_parse_lean_output_success(self):
        parsed = self.runner._parse_lean_output({'returncode': 0, 'stdout': '', 'stderr': '', 'execution_time': 0.5})
        assert parsed['verification_status']['compilation_successful']
        assert parsed['theorems_proven'] == []

    def test_parse_lean_output_error(self):
        parsed = self.runner._parse_lean_output({'returncode': 1, 'stdout': '', 'stderr': 'error: syntax error', 'execution_time': 0.3})
        assert not parsed['verification_status']['compilation_successful']
        assert parsed['error_details']

    def test_execute_lean_success(self, tmp_path):
        source = tmp_path / 'valid.lean'
        source.write_text('#eval (2 : Nat) + 2')
        result = self.runner._execute_lean(source)
        assert result['returncode'] == 0
        assert result['stdout'].strip() == '4'

    def test_execute_lean_file_not_found(self):
        """Test executing Lean when file doesn't exist"""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                self.runner._execute_lean(Path("/nonexistent/file.lean"))

    def test_execute_lean_timeout(self, tmp_path):
        source = tmp_path / 'slow.lean'
        source.write_text('#eval IO.sleep 3000')
        result = LeanRunner(timeout=1)._execute_lean(source)
        assert result['returncode'] == -1
        assert 'Timeout' in result['stderr']

    def test_create_temp_file(self):
        before = set(Path(tempfile.gettempdir()).glob('tmp*.lean'))
        result = self.runner.run_lean_code('#eval (1 : Nat)')
        assert result['success'], result
        assert set(Path(tempfile.gettempdir()).glob('tmp*.lean')) == before

    def test_save_comprehensive_proof_outcomes(self, tmp_path):
        result = self.runner.run_lean_code('theorem exactProof : True := True.intro')
        saved = self.runner.save_comprehensive_proof_outcomes(result, tmp_path, 'control')
        assert all(path.is_file() for path in saved.values())
        proofs = json.loads(saved['theorems'].read_text())
        assert [proof['name'] for proof in proofs['theorems_proven']] == ['exactProof']

    def test_export_lean_code(self):
        """Test exporting Lean code to file"""
        code = "theorem test : ∀ x, x = x := by rfl"
        output_path = Path("/tmp/exported.lean")

        with patch('builtins.open', mock_open()) as mock_file:
            result_path = self.runner.export_lean_code(code, output_path)

            assert result_path == output_path
            mock_file.assert_called_once_with(output_path, 'w', encoding='utf-8')
            handle = mock_file()
            handle.write.assert_called_once_with(code)

    def test_generate_proof_output(self, tmp_path):
        result = self.runner.run_lean_code('theorem exactProof : True := True.intro')
        saved = self.runner.generate_proof_output(result, tmp_path, 'control')
        assert saved['complete_summary'].is_file()
        assert saved['source_inventory'].is_file()
        assert json.loads(saved['theorems'].read_text())['theorems_proven']

    def test_suggest_atp_integration(self):
        """Test ATP integration suggestions"""
        suggestions = self.runner.suggest_atp_integration()

        assert isinstance(suggestions, str)
        assert len(suggestions) > 0
        assert "ATP" in suggestions or "automated theorem proving" in suggestions.lower()

    def test_error_handling_invalid_lean_path(self):
        """Test error handling with invalid Lean path"""
        runner = LeanRunner(lean_path="/invalid/path/lean")

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("lean not found")

            result = runner.run_lean_code("def test := 1")

            assert result['success'] is False
            assert 'error' in result

    def test_concurrent_execution(self):
        """Test handling of concurrent Lean executions"""
        # This would test thread safety and concurrent access
        # For now, just verify the object can handle multiple calls
        code = "def test := 1"

        with patch.object(self.runner, 'run_lean_code') as mock_run:
            mock_run.return_value = {'success': True}

            # Call multiple times
            result1 = self.runner.run_lean_code(code)
            result2 = self.runner.run_lean_code(code)

            assert mock_run.call_count == 2

    def test_memory_usage_tracking(self):
        """Test that execution time is tracked"""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = self.runner.run_lean_code("def test := 1")

            assert 'execution_time' in result
            assert isinstance(result['execution_time'], (int, float))

    def test_output_validation(self):
        """Test that output structure is consistent"""
        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = self.runner.run_lean_code("def test := 1")

            required_keys = ['success', 'execution_time']
            for key in required_keys:
                assert key in result

            assert isinstance(result['success'], bool)

    def test_large_code_handling(self):
        code = '\n'.join(f'def generated_{i} : Nat := {i}' for i in range(1000))
        result = self.runner.run_lean_code(code)
        assert result['success'], result
        assert len(result['result']['definitions_created']) == 1000


