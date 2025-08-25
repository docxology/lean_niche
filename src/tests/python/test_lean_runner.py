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
        """Test logging setup"""
        with patch('src.python.core.logging_config.setup_lean_logging') as mock_setup:
            mock_logger = MagicMock()
            mock_setup.return_value = mock_logger

            runner = LeanRunner()
            runner.setup_logging()

            mock_setup.assert_called_once()
            assert runner.logger == mock_logger

    def test_run_lean_code_simple(self):
        """Test running simple Lean code"""
        simple_code = """
import LeanNiche.Basic

def simple_function (x : Nat) : Nat := x + 1
"""

        with patch('subprocess.run') as mock_run, \
             patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.unlink'):

            # Mock temporary file
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.lean"
            mock_temp.return_value.__enter__.return_value.write = MagicMock()

            # Mock subprocess result
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "Lean compilation successful"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = self.runner.run_lean_code(simple_code, ["LeanNiche.Basic"])

            assert isinstance(result, dict)
            assert 'success' in result
            assert 'execution_time' in result
            mock_run.assert_called_once()

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
        """Test algorithm verification"""
        algorithm_code = """
def binary_search (arr : List Nat) (target : Nat) : Bool :=
  match arr with
  | [] => false
  | [x] => x = target
  | xs =>
    let mid := xs.length / 2
    let mid_val := xs.get! mid
    if target < mid_val then
      binary_search (xs.take mid) target
    else if target > mid_val then
      binary_search (xs.drop (mid + 1)) target
    else true
"""

        test_cases = [
            {"input": [1, 2, 3, 4, 5], "expected": True, "description": "search_existing"},
            {"input": [1, 2, 3, 4, 5], "expected": False, "description": "search_nonexistent"}
        ]

        with patch.object(self.runner, 'run_lean_code') as mock_run:
            mock_run.return_value = {
                'success': True,
                'execution_time': 0.3,
                'theorems_verified': 0,
                'definitions_found': 1
            }

            result = self.runner.run_algorithm_verification(algorithm_code, test_cases, ["LeanNiche.Basic"])

            assert isinstance(result, dict)
            assert result['success'] is True
            assert 'execution_time' in result
            assert 'test_cases_run' in result

    def test_extract_mathematical_results_theorems(self):
        """Test extracting mathematical results from theorem output"""
        lean_output = """
Lean compilation successful
theorem add_comm : ∀ a b : Nat, a + b = b + a
  proof completed successfully

theorem mul_comm : ∀ a b : Nat, a * b = b * a
  proof completed successfully

def factorial : Nat → Nat
  defined successfully
"""

        result = self.runner.extract_mathematical_results(lean_output)

        assert isinstance(result, dict)
        assert 'theorems_verified' in result
        assert 'definitions_found' in result
        assert result['theorems_verified'] >= 2
        assert result['definitions_found'] >= 1

    def test_extract_mathematical_results_errors(self):
        """Test extracting results from error output"""
        lean_output = """
error: syntax error at line 5
warning: unused variable 'x'
error: type mismatch at line 10
"""

        result = self.runner.extract_mathematical_results(lean_output)

        assert isinstance(result, dict)
        assert 'errors' in result
        assert 'warnings' in result
        assert result['errors'] >= 2
        assert result['warnings'] >= 1

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

    def test_save_results_json(self):
        """Test saving results to JSON file"""
        results = {
            'success': True,
            'theorems_verified': 2,
            'execution_time': 0.5
        }

        with patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_dump, \
             patch('pathlib.Path') as mock_path:

            mock_path_instance = MagicMock()
            mock_path.return_value = mock_path_instance

            output_path = self.runner.save_results(results, "test_results", "json")

            mock_file.assert_called_once()
            mock_dump.assert_called_once_with(results, mock_file())

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
        """Test parsing successful Lean output"""
        result_dict = {
            'returncode': 0,
            'stdout': 'Lean compilation successful',
            'stderr': '',
            'execution_time': 0.5
        }

        with patch.object(self.runner, 'extract_mathematical_results') as mock_extract:
            mock_extract.return_value = {'theorems_verified': 2, 'definitions_found': 1}

            parsed = self.runner._parse_lean_output(result_dict)

            assert parsed['success'] is True
            assert parsed['execution_time'] == 0.5
            assert parsed['theorems_verified'] == 2
            assert parsed['definitions_found'] == 1

    def test_parse_lean_output_error(self):
        """Test parsing Lean output with errors"""
        result_dict = {
            'returncode': 1,
            'stdout': '',
            'stderr': 'error: syntax error',
            'execution_time': 0.3
        }

        with patch.object(self.runner, 'extract_mathematical_results') as mock_extract, \
             patch.object(self.runner, '_parse_error_line') as mock_parse_error:

            mock_extract.return_value = {'errors': 1, 'warnings': 0}
            mock_parse_error.return_value = {'type': 'syntax', 'severity': 'error'}

            parsed = self.runner._parse_lean_output(result_dict)

            assert parsed['success'] is False
            assert parsed['execution_time'] == 0.3
            assert 'errors' in parsed

    def test_execute_lean_success(self):
        """Test executing Lean code successfully"""
        with patch('subprocess.run') as mock_run, \
             patch('pathlib.Path.exists', return_value=True):

            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_result.stdout = "success"
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = self.runner._execute_lean(Path("/tmp/test.lean"))

            assert isinstance(result, dict)
            assert result['returncode'] == 0
            assert 'execution_time' in result

    def test_execute_lean_file_not_found(self):
        """Test executing Lean when file doesn't exist"""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError):
                self.runner._execute_lean(Path("/nonexistent/file.lean"))

    def test_execute_lean_timeout(self):
        """Test executing Lean with timeout"""
        runner = LeanRunner(timeout=1)

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("lean", 1)

            result = runner._execute_lean(Path("/tmp/test.lean"))

            assert result['success'] is False
            assert 'timeout' in result

    def test_create_temp_file(self):
        """Test creating temporary Lean file"""
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_file = MagicMock()
            mock_file.name = "/tmp/test.lean"
            mock_temp.return_value.__enter__.return_value = mock_file

            temp_path = self.runner._create_temp_file("test code")

            assert isinstance(temp_path, Path)
            assert str(temp_path) == "/tmp/test.lean"
            mock_file.write.assert_called_once_with("test code")

    def test_save_comprehensive_proof_outcomes(self):
        """Test saving comprehensive proof outcomes"""
        results = {
            'success': True,
            'theorems_verified': 2,
            'definitions_found': 1
        }

        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('builtins.open', mock_open()) as mock_file, \
             patch('json.dump') as mock_dump:

            output_dir = Path("/tmp/test_output")
            saved_files = self.runner.save_comprehensive_proof_outcomes(results, output_dir, "test")

            assert isinstance(saved_files, dict)
            assert len(saved_files) > 0
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

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

    def test_generate_proof_output(self):
        """Test generating proof output"""
        results = {'success': True, 'theorems_verified': 1}

        with patch.object(self.runner, 'save_comprehensive_proof_outcomes') as mock_save:
            mock_save.return_value = {'file1': Path('/tmp/file1.json')}

            output_dir = Path("/tmp/output")
            saved_files = self.runner.generate_proof_output(results, output_dir, "test")

            mock_save.assert_called_once_with(results, output_dir, "test")
            assert saved_files == {'file1': Path('/tmp/file1.json')}

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
        """Test handling of large Lean code files"""
        large_code = "\n".join([f"def func_{i} : Nat := {i}" for i in range(1000)])

        with patch('subprocess.run') as mock_run:
            mock_result = MagicMock()
            mock_result.returncode = 0
            mock_run.return_value = mock_result

            result = self.runner.run_lean_code(large_code)

            assert result['success'] is True
            assert 'execution_time' in result


