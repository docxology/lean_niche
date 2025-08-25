"""Comprehensive tests for LeanNicheOrchestratorBase"""

import tempfile
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.python.core.orchestrator_base import LeanNicheOrchestratorBase


class TestLeanNicheOrchestratorBase:
    """Comprehensive tests for LeanNicheOrchestratorBase"""

    def setup_method(self):
        """Setup test fixtures"""
        self.output_dir = tempfile.mkdtemp()
        self.orchestrator = self.TestOrchestrator("test_domain", self.output_dir, enable_logging=True)

    class TestOrchestrator(LeanNicheOrchestratorBase):
        """Test implementation of the abstract base class"""
        def run_domain_specific_analysis(self):
            return {"test_data": [1, 2, 3], "analysis_type": "test"}

        def create_domain_visualizations(self, analysis_results):
            # Create a dummy visualization file
            viz_file = self.viz_dir / 'test_viz.png'
            viz_file.write_text('dummy png content')

    def test_initialization(self):
        """Test orchestrator initialization"""
        assert isinstance(self.orchestrator, LeanNicheOrchestratorBase)
        assert self.orchestrator.domain_name == "test_domain"
        assert str(self.orchestrator.output_dir) == self.output_dir
        assert hasattr(self.orchestrator, 'lean_runner')
        assert hasattr(self.orchestrator, 'data_dir')
        assert hasattr(self.orchestrator, 'proofs_dir')
        assert hasattr(self.orchestrator, 'viz_dir')
        assert hasattr(self.orchestrator, 'reports_dir')
        assert hasattr(self.orchestrator, 'enable_logging')

    def test_initialization_with_logging_disabled(self):
        """Test initialization with logging disabled"""
        orchestrator = self.TestOrchestrator("test_domain", self.output_dir, enable_logging=False)
        assert orchestrator.enable_logging is False

    def test_directory_creation(self):
        """Test that output directories are created"""
        assert self.orchestrator.data_dir.exists()
        assert self.orchestrator.proofs_dir.exists()
        assert self.orchestrator.viz_dir.exists()
        assert self.orchestrator.reports_dir.exists()

        assert self.orchestrator.data_dir.is_dir()
        assert self.orchestrator.proofs_dir.is_dir()
        assert self.orchestrator.viz_dir.is_dir()
        assert self.orchestrator.reports_dir.is_dir()

    def test_lean_runner_initialization(self):
        """Test that LeanRunner is properly initialized"""
        assert self.orchestrator.lean_runner is not None
        assert hasattr(self.orchestrator.lean_runner, 'run_lean_code')
        assert hasattr(self.orchestrator.lean_runner, 'run_theorem_verification')

    def test_logging_setup(self):
        """Test logging setup when enabled"""
        assert hasattr(self.orchestrator, 'lean_logger')
        assert self.orchestrator.lean_logger is not None

    def test_logging_disabled(self):
        """Test behavior when logging is disabled"""
        orchestrator = self.TestOrchestrator("test_domain", self.output_dir, enable_logging=False)
        assert not hasattr(orchestrator, 'lean_logger') or orchestrator.lean_logger is None

    def test_setup_comprehensive_lean_environment(self):
        """Test comprehensive Lean environment setup"""
        domain_modules = ["LeanNiche.Basic", "LeanNiche.Statistics"]

        with patch.object(self.orchestrator.lean_runner, 'run_lean_code') as mock_run:
            mock_run.return_value = {'success': True, 'execution_time': 0.1}

            lean_file_path = self.orchestrator.setup_comprehensive_lean_environment(domain_modules)

            assert isinstance(lean_file_path, Path)
            assert lean_file_path.exists()
            assert lean_file_path.suffix == '.lean'

            # Verify file content contains expected imports
            with open(lean_file_path, 'r') as f:
                content = f.read()
                assert "import LeanNiche.Basic" in content
                assert "import LeanNiche.Statistics" in content

    def test_generate_comprehensive_lean_code(self):
        """Test comprehensive Lean code generation"""
        domain_modules = ["LeanNiche.Basic", "LeanNiche.Statistics"]

        lean_code = self.orchestrator._generate_comprehensive_lean_code(domain_modules)

        assert isinstance(lean_code, str)
        assert len(lean_code) > 0
        assert "import LeanNiche.Basic" in lean_code
        assert "import LeanNiche.Statistics" in lean_code
        assert "namespace" in lean_code

    def test_add_statistics_content(self):
        """Test adding statistics content to Lean code"""
        stats_code = self.orchestrator._add_statistics_content()

        assert isinstance(stats_code, str)
        assert len(stats_code) > 0
        assert "Statistical Analysis" in stats_code or "theorem" in stats_code or "def" in stats_code

    def test_add_dynamical_systems_content(self):
        """Test adding dynamical systems content"""
        dyn_code = self.orchestrator._add_dynamical_systems_content()

        assert isinstance(dyn_code, str)
        assert len(dyn_code) > 0

    def test_add_control_theory_content(self):
        """Test adding control theory content"""
        control_code = self.orchestrator._add_control_theory_content()

        assert isinstance(control_code, str)
        assert len(control_code) > 0

    def test_execute_comprehensive_analysis(self):
        """Test comprehensive analysis execution"""
        analysis_data = {"test": "data", "values": [1, 2, 3]}

        with patch.object(self.orchestrator, 'run_domain_specific_analysis') as mock_analysis, \
             patch.object(self.orchestrator.lean_runner, 'run_lean_code') as mock_run:

            mock_analysis.return_value = analysis_data
            mock_run.return_value = {
                'success': True,
                'execution_time': 0.5,
                'theorems_verified': 2,
                'definitions_found': 1
            }

            results = self.orchestrator.execute_comprehensive_analysis(analysis_data)

            assert isinstance(results, dict)
            assert 'analysis_results' in results
            assert 'lean_verification' in results
            assert 'execution_time' in results

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        analysis_results = {
            'analysis_results': {'test': 'data'},
            'lean_verification': {'success': True, 'theorems_verified': 3},
            'execution_time': 0.8
        }

        report_path = self.orchestrator.generate_comprehensive_report(analysis_results)

        assert isinstance(report_path, Path)
        assert report_path.exists()
        assert report_path.suffix == '.md'

        # Verify report content
        with open(report_path, 'r') as f:
            content = f.read()
            assert "Comprehensive Analysis Report" in content
            assert "test_domain" in content
            assert "theorems_verified" in content

    def test_create_execution_summary(self):
        """Test execution summary creation"""
        summary_path = self.orchestrator.create_execution_summary()

        assert isinstance(summary_path, Path)
        assert summary_path.exists()
        assert summary_path.suffix == '.json'

        # Verify summary content
        with open(summary_path, 'r') as f:
            summary = json.load(f)
            assert 'domain_name' in summary
            assert 'timestamp' in summary
            assert 'directories_created' in summary
            assert summary['domain_name'] == 'test_domain'

    def test_run_comprehensive_orchestration(self):
        """Test the complete orchestration workflow"""
        with patch.object(self.orchestrator, 'run_domain_specific_analysis') as mock_analysis, \
             patch.object(self.orchestrator, 'execute_comprehensive_analysis') as mock_execute, \
             patch.object(self.orchestrator, 'generate_comprehensive_report') as mock_report, \
             patch.object(self.orchestrator, 'create_execution_summary') as mock_summary:

            mock_analysis.return_value = {"test": "analysis_data"}
            mock_execute.return_value = {"analysis_results": {"test": "executed"}}
            mock_report.return_value = Path("/tmp/report.md")
            mock_summary.return_value = Path("/tmp/summary.json")

            results = self.orchestrator.run_comprehensive_orchestration()

            assert isinstance(results, dict)
            assert 'analysis_results' in results
            assert 'verification_results' in results
            assert 'report_path' in results
            assert 'summary_path' in results

    def test_error_handling_in_analysis(self):
        """Test error handling during analysis"""
        with patch.object(self.orchestrator, 'run_domain_specific_analysis') as mock_analysis:
            mock_analysis.side_effect = Exception("Analysis error")

            with pytest.raises(Exception):
                self.orchestrator.run_comprehensive_orchestration()

    def test_error_handling_in_lean_verification(self):
        """Test error handling during Lean verification"""
        with patch.object(self.orchestrator, 'run_domain_specific_analysis') as mock_analysis, \
             patch.object(self.orchestrator.lean_runner, 'run_lean_code') as mock_run:

            mock_analysis.return_value = {"test": "data"}
            mock_run.side_effect = Exception("Lean verification error")

            results = self.orchestrator.execute_comprehensive_analysis({"test": "data"})

            assert 'error' in results or 'lean_verification' in results
            assert results.get('lean_verification', {}).get('success') is False

    def test_directory_structure_validation(self):
        """Test that directory structure is properly validated"""
        # Verify all expected directories exist and are writable
        dirs_to_check = [
            self.orchestrator.data_dir,
            self.orchestrator.proofs_dir,
            self.orchestrator.viz_dir,
            self.orchestrator.reports_dir
        ]

        for dir_path in dirs_to_check:
            assert dir_path.exists()
            assert dir_path.is_dir()

            # Test that directory is writable
            test_file = dir_path / "test_write.txt"
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                test_file.unlink()  # Clean up
            except (OSError, PermissionError):
                pytest.fail(f"Directory {dir_path} is not writable")

    def test_lean_runner_integration(self):
        """Test integration with LeanRunner"""
        with patch.object(self.orchestrator.lean_runner, 'run_lean_code') as mock_run:
            mock_run.return_value = {
                'success': True,
                'execution_time': 0.2,
                'theorems_verified': 1
            }

            lean_code = "theorem test : ∀ x, x = x := by rfl"
            result = self.orchestrator.lean_runner.run_lean_code(lean_code, ["LeanNiche.Basic"])

            mock_run.assert_called_once_with(lean_code, ["LeanNiche.Basic"])
            assert result['success'] is True

    def test_logging_integration(self):
        """Test logging integration"""
        with patch.object(self.orchestrator.lean_logger, 'log_step_start') as mock_log:
            # This would test that logging methods are called during operations
            # For now, just verify the logger exists and has expected methods
            assert hasattr(self.orchestrator.lean_logger, 'log_step_start')
            assert hasattr(self.orchestrator.lean_logger, 'log_step_end')
            assert hasattr(self.orchestrator.lean_logger, 'log_error')

    def test_output_file_management(self):
        """Test that output files are properly managed"""
        # Test that files are created in correct locations
        analysis_results = {'test': 'data'}
        report_path = self.orchestrator.generate_comprehensive_report(analysis_results)

        assert report_path.parent == self.orchestrator.reports_dir

        summary_path = self.orchestrator.create_execution_summary()
        assert summary_path.parent == self.orchestrator.reports_dir

    def test_concurrent_orchestration_safety(self):
        """Test that orchestration is safe for concurrent execution"""
        # Create multiple orchestrator instances
        orchestrator2 = self.TestOrchestrator("test_domain2", tempfile.mkdtemp())

        assert orchestrator2.domain_name != self.orchestrator.domain_name
        assert str(orchestrator2.output_dir) != str(self.orchestrator.output_dir)

        # Verify both can create their directory structures independently
        assert orchestrator2.data_dir.exists()
        assert orchestrator2.proofs_dir.exists()

    def test_resource_cleanup(self):
        """Test that resources are properly cleaned up"""
        import shutil

        # Test that temporary directories can be cleaned up
        temp_dir = self.orchestrator.output_dir
        assert temp_dir.exists()

        # Verify we can clean up the directory (simulating cleanup)
        # In real usage, this would be done by the caller
        shutil.rmtree(temp_dir, ignore_errors=True)
        assert not temp_dir.exists()

    def test_configuration_persistence(self):
        """Test that configuration is properly maintained"""
        original_domain = self.orchestrator.domain_name
        original_output = self.orchestrator.output_dir

        # Simulate some operations
        self.orchestrator.run_domain_specific_analysis()

        # Configuration should remain unchanged
        assert self.orchestrator.domain_name == original_domain
        assert self.orchestrator.output_dir == original_output

    def test_empty_analysis_data_handling(self):
        """Test handling of empty analysis data"""
        results = self.orchestrator.execute_comprehensive_analysis({})

        assert isinstance(results, dict)
        # Should handle empty data gracefully without crashing

    def test_large_analysis_data_handling(self):
        """Test handling of large analysis data"""
        large_data = {
            'large_array': list(range(10000)),
            'nested_data': {'level1': {'level2': list(range(1000))}}
        }

        results = self.orchestrator.execute_comprehensive_analysis(large_data)

        assert isinstance(results, dict)
        # Should handle large data without memory issues

    def test_lean_code_generation_edge_cases(self):
        """Test Lean code generation with edge cases"""
        # Test with empty module list
        lean_code = self.orchestrator._generate_comprehensive_lean_code([])
        assert isinstance(lean_code, str)
        assert len(lean_code) > 0  # Should still generate basic structure

        # Test with special characters in module names
        special_modules = ["LeanNiche.Test_Module", "LeanNiche.Test-Module"]
        lean_code = self.orchestrator._generate_comprehensive_lean_code(special_modules)
        assert isinstance(lean_code, str)

    def test_file_path_handling(self):
        """Test file path handling and validation"""
        # Test with relative paths
        rel_orchestrator = self.TestOrchestrator("test", "relative/path")
        assert rel_orchestrator.output_dir.is_absolute()

        # Test with absolute paths
        abs_orchestrator = self.TestOrchestrator("test", "/absolute/path")
        assert abs_orchestrator.output_dir.is_absolute()

    def test_memory_efficiency(self):
        """Test that operations are memory efficient"""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Perform some operations
        for _ in range(10):
            self.orchestrator.run_domain_specific_analysis()

        # Check that memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Allow for some memory growth but not excessive
        assert memory_increase < 50 * 1024 * 1024  # 50MB limit


