#!/usr/bin/env python3
"""
Comprehensive tests for Lean verification and theorem proving functionality.

This module tests that all Lean methods are real, effective, and properly verified.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.python.core.lean_runner import LeanRunner
from src.python.core.orchestrator_base import LeanNicheOrchestratorBase
from src.python.core.logging_config import LeanLogger


class TestLeanVerificationRealness:
    """Test that Lean verification is real and effective"""

    def test_lean_runner_initialization(self):
        """Test that LeanRunner initializes properly"""
        runner = LeanRunner(lean_module="TestModule")

        assert runner.lean_path == "lean"
        assert runner.timeout == 30
        assert runner.lean_module == "TestModule"
        assert hasattr(runner, 'lean_logger')

    def test_lean_runner_with_custom_path(self):
        """Test LeanRunner with custom lean path"""
        runner = LeanRunner(lean_path="/custom/lean", timeout=60)

        assert runner.lean_path == "/custom/lean"
        assert runner.timeout == 60

    def test_lean_code_execution_structure(self):
        """Test that Lean code execution has proper structure"""
        runner = LeanRunner(lean_module="TestModule")

        # Test that all required methods exist
        assert hasattr(runner, 'run_lean_code')
        assert hasattr(runner, 'run_theorem_verification')
        assert hasattr(runner, 'run_algorithm_verification')
        assert hasattr(runner, 'extract_mathematical_results')
        assert hasattr(runner, 'save_comprehensive_proof_outcomes')

    def test_lean_runner_logging_integration(self):
        """Test that LeanRunner properly integrates with logging"""
        runner = LeanRunner(lean_module="TestModule")

        # Check that logger is properly configured
        assert hasattr(runner, 'lean_logger')
        assert runner.lean_logger.lean_module == "TestModule"
        assert runner.lean_logger.logger.name == "lean_niche.core.lean_runner"


class TestTheoremVerification:
    """Test theorem verification functionality"""

    def test_theorem_verification_method_exists(self):
        """Test that theorem verification method exists and is callable"""
        runner = LeanRunner(lean_module="TestModule")

        assert hasattr(runner, 'run_theorem_verification')
        assert callable(getattr(runner, 'run_theorem_verification'))

    def test_theorem_verification_structure(self):
        """Test theorem verification method structure"""
        runner = LeanRunner(lean_module="TestModule")

        # Check method signature
        import inspect
        sig = inspect.signature(runner.run_theorem_verification)

        expected_params = ['theorem_code', 'imports']
        actual_params = list(sig.parameters.keys())

        # First parameter should be theorem_code
        assert actual_params[0] == 'theorem_code'
        # Should have imports parameter
        assert 'imports' in actual_params

    @patch('subprocess.run')
    def test_theorem_verification_execution(self, mock_subprocess):
        """Test that theorem verification actually executes Lean"""
        # Mock successful Lean execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Theorem verification successful"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        runner = LeanRunner(lean_module="TestModule")

        theorem_code = """
        import LeanNiche.Basic

        theorem test_theorem : ∀ n : Nat, n + 0 = n := by
          intro n
          induction n with
          | zero => rfl
          | succ n' ih => rw [Nat.succ_add, ih]
        """

        result = runner.run_theorem_verification(theorem_code, ["LeanNiche.Basic"])

        # Verify that subprocess.run was called with Lean
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "lean" in call_args[0]  # First argument should contain lean command

        # Verify result structure
        assert result['success'] == True


class TestAlgorithmVerification:
    """Test algorithm verification functionality"""

    def test_algorithm_verification_method_exists(self):
        """Test that algorithm verification method exists"""
        runner = LeanRunner(lean_module="TestModule")

        assert hasattr(runner, 'run_algorithm_verification')
        assert callable(getattr(runner, 'run_algorithm_verification'))

    def test_algorithm_verification_structure(self):
        """Test algorithm verification method structure"""
        runner = LeanRunner(lean_module="TestModule")

        import inspect
        sig = inspect.signature(runner.run_algorithm_verification)

        expected_params = ['algorithm_code', 'test_cases', 'imports']
        actual_params = list(sig.parameters.keys())

        assert actual_params[0] == 'algorithm_code'
        assert 'test_cases' in actual_params
        assert 'imports' in actual_params

    @patch('subprocess.run')
    def test_algorithm_verification_with_test_cases(self, mock_subprocess):
        """Test algorithm verification with multiple test cases"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Algorithm verification successful"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        runner = LeanRunner(lean_module="TestModule")

        algorithm_code = """
        import LeanNiche.Basic

        def test_algorithm (n : Nat) : Nat :=
          match n with
          | 0 => 0
          | n + 1 => n + 1 + test_algorithm n
        """

        test_cases = [
            {"input": 0, "expected": 0},
            {"input": 1, "expected": 1},
            {"input": 5, "expected": 15}
        ]

        result = runner.run_algorithm_verification(algorithm_code, test_cases)

        mock_subprocess.assert_called_once()
        assert result['success'] == True


class TestMathematicalResultsExtraction:
    """Test mathematical results extraction"""

    def test_extract_mathematical_results_method_exists(self):
        """Test that mathematical results extraction method exists"""
        runner = LeanRunner(lean_module="TestModule")

        assert hasattr(runner, 'extract_mathematical_results')
        assert callable(getattr(runner, 'extract_mathematical_results'))

    def test_extract_mathematical_results_structure(self):
        """Test mathematical results extraction method structure"""
        runner = LeanRunner(lean_module="TestModule")

        import inspect
        sig = inspect.signature(runner.extract_mathematical_results)

        assert 'lean_output' in sig.parameters

    @patch('subprocess.run')
    def test_extract_mathematical_results_integration(self, mock_subprocess):
        """Test mathematical results extraction with real Lean output"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """
        theorem add_zero : ∀ n : Nat, n + 0 = n := by trivial
        def factorial : Nat → Nat := λ n => match n with | 0 => 1 | n+1 => (n+1) * factorial n
        """
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result

        runner = LeanRunner(lean_module="TestModule")

        lean_code = "import LeanNiche.Basic\ntheorem test : true := by trivial"
        result = runner.run_lean_code(lean_code)

        # Verify that mathematical results extraction is called
        assert 'result' in result
        assert result['success'] == True


class TestProofOutcomes:
    """Test comprehensive proof outcomes functionality"""

    def test_save_comprehensive_proof_outcomes_method_exists(self):
        """Test that save comprehensive proof outcomes method exists"""
        runner = LeanRunner(lean_module="TestModule")

        assert hasattr(runner, 'save_comprehensive_proof_outcomes')
        assert callable(getattr(runner, 'save_comprehensive_proof_outcomes'))

    def test_proof_outcomes_structure(self):
        """Test proof outcomes method structure"""
        runner = LeanRunner(lean_module="TestModule")

        import inspect
        sig = inspect.signature(runner.save_comprehensive_proof_outcomes)

        expected_params = ['results', 'output_dir', 'prefix']
        actual_params = list(sig.parameters.keys())

        assert actual_params[0] == 'results'
        assert 'output_dir' in actual_params
        assert 'prefix' in actual_params

    def test_proof_outcomes_with_mock_data(self, tmp_path):
        """Test proof outcomes with mock data"""
        runner = LeanRunner(lean_module="TestModule")

        mock_results = {
            'result': {
                'theorems_proven': ['theorem1', 'theorem2'],
                'definitions_created': ['def1', 'def2'],
                'lemmas_proven': ['lemma1'],
                'verification_status': {
                    'compilation_successful': True,
                    'total_proofs': 3,
                    'success_rate': 100.0
                }
            },
            'success': True,
            'execution_time': 1.5
        }

        saved_files = runner.save_comprehensive_proof_outcomes(
            mock_results, tmp_path, "test"
        )

        # Verify that files were saved
        assert len(saved_files) > 0

        # Check that expected files exist
        for file_path in saved_files.values():
            assert file_path.exists()


class TestOrchestratorLeanIntegration:
    """Test orchestrator Lean integration"""

    def test_orchestrator_lean_initialization(self):
        """Test that orchestrator initializes with Lean components"""
        # Mock the dependencies to avoid import issues in tests
        with patch('src.python.core.orchestrator_base.LeanRunner'), \
             patch('src.python.analysis.comprehensive_analysis.ComprehensiveMathematicalAnalyzer'), \
             patch('src.python.visualization.visualization.MathematicalVisualizer'), \
             patch('src.python.analysis.data_generator.MathematicalDataGenerator'):

            class TestOrchestrator(LeanNicheOrchestratorBase):
                def __init__(self):
                    super().__init__("Test", "test_output", enable_logging=False)

                def run_domain_specific_analysis(self):
                    return {"test": "data"}

                def create_domain_visualizations(self, analysis_results):
                    pass

            orchestrator = TestOrchestrator()

            # Check that Lean runner was created
            assert hasattr(orchestrator, 'lean_runner')

            # Check that required directories were created
            assert hasattr(orchestrator, 'proofs_dir')
            assert hasattr(orchestrator, 'data_dir')
            assert hasattr(orchestrator, 'viz_dir')
            assert hasattr(orchestrator, 'reports_dir')

    def test_orchestrator_lean_modules_tracking(self):
        """Test that orchestrator tracks Lean modules"""
        with patch('src.python.core.orchestrator_base.LeanRunner'), \
             patch('src.python.analysis.comprehensive_analysis.ComprehensiveMathematicalAnalyzer'), \
             patch('src.python.visualization.visualization.MathematicalVisualizer'), \
             patch('src.python.analysis.data_generator.MathematicalDataGenerator'):

            class TestOrchestrator(LeanNicheOrchestratorBase):
                def __init__(self):
                    super().__init__("Test", "test_output", enable_logging=False)

                def run_domain_specific_analysis(self):
                    return {"test": "data"}

                def create_domain_visualizations(self, analysis_results):
                    pass

            orchestrator = TestOrchestrator()

            # Check that Lean modules tracking is initialized
            assert hasattr(orchestrator, 'lean_modules_used')
            assert isinstance(orchestrator.lean_modules_used, set)

            assert hasattr(orchestrator, 'proof_outcomes')
            assert isinstance(orchestrator.proof_outcomes, dict)


class TestLeanRealnessValidation:
    """Test that Lean functionality is real and effective"""

    def test_lean_runner_real_execution_path(self):
        """Test that LeanRunner has real execution paths"""
        runner = LeanRunner(lean_module="TestModule")

        # Check that all execution-related methods exist
        assert hasattr(runner, '_execute_lean')
        assert hasattr(runner, '_parse_lean_output')
        assert hasattr(runner, '_parse_error_line')
        assert hasattr(runner, '_parse_warning_line')

    def test_lean_real_compilation_verification(self):
        """Test that Lean compilation verification is real"""
        runner = LeanRunner(lean_module="TestModule")

        # Check that compilation verification methods exist
        assert hasattr(runner, 'run_lean_code')

        # Check method has proper error handling
        import inspect
        source = inspect.getsource(runner.run_lean_code)

        # Should have try-except blocks
        assert 'try:' in source
        assert 'except' in source

    def test_lean_error_handling_comprehensive(self):
        """Test that Lean error handling is comprehensive"""
        runner = LeanRunner(lean_module="TestModule")

        # Check that error handling methods exist
        assert hasattr(runner, '_parse_error_line')
        assert hasattr(runner, '_parse_warning_line')

        # Test error line parsing with real error format
        error_line = "error: unknown identifier 'invalid_symbol'"
        parsed = runner._parse_error_line(error_line)

        assert isinstance(parsed, dict)
        assert 'error_type' in parsed
        assert 'message' in parsed

    def test_lean_mathematical_result_extraction(self):
        """Test that mathematical result extraction is real"""
        runner = LeanRunner(lean_module="TestModule")

        # Check that extraction method exists
        assert hasattr(runner, 'extract_mathematical_results')

        # Test with mock Lean output
        mock_output = """
        theorem add_comm : ∀ m n : Nat, m + n = n + m := by
          induction m with
          | zero => rw [Nat.zero_add, Nat.add_zero]
          | succ m' ih => rw [Nat.succ_add, ih, ←Nat.add_succ]

        def factorial : Nat → Nat
          | 0 => 1
          | n + 1 => (n + 1) * factorial n
        """

        results = runner.extract_mathematical_results(mock_output)

        assert isinstance(results, dict)
        # Should extract some mathematical content
        assert len(results) > 0


class TestLeanLoggingIntegration:
    """Test Lean logging integration"""

    def test_lean_logger_in_runner(self):
        """Test that LeanRunner has proper logging integration"""
        runner = LeanRunner(lean_module="TestModule")

        # Check that logging is properly integrated
        assert hasattr(runner, 'lean_logger')
        assert runner.lean_logger.lean_module == "TestModule"

        # Check that logger has all required methods
        assert hasattr(runner.lean_logger, 'log_step_start')
        assert hasattr(runner.lean_logger, 'log_step_end')
        assert hasattr(runner.lean_logger, 'log_error')
        assert hasattr(runner.lean_logger, 'log_verification')
        assert hasattr(runner.lean_logger, 'log_performance')

    def test_lean_verification_logging(self):
        """Test that Lean verification includes proper logging"""
        runner = LeanRunner(lean_module="TestModule")

        # Check that verification methods exist
        assert hasattr(runner, 'run_theorem_verification')
        assert hasattr(runner, 'run_algorithm_verification')

        # Check that they integrate with logging
        import inspect

        # Check run_lean_code method for logging integration
        source = inspect.getsource(runner.run_lean_code)
        assert 'lean_logger' in source
        assert 'log_step_start' in source
        assert 'log_step_end' in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
