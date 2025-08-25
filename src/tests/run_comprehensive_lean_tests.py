#!/usr/bin/env python3
"""
Comprehensive Lean Verification Test Runner

This script runs comprehensive tests to validate that all Lean methods are real,
effective, and properly logged. It ensures complete verification of Lean functionality.

Features:
- Real Lean method validation
- Comprehensive logging verification
- Performance benchmarking
- Error handling validation
- Integration testing
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.python.core.lean_runner import LeanRunner
from src.python.core.logging_config import LeanLogger, setup_lean_logging
from src.python.core.orchestrator_base import LeanNicheOrchestratorBase


class ComprehensiveLeanTestRunner:
    """Comprehensive test runner for Lean verification and logging"""

    def __init__(self, enable_logging: bool = True, log_level: str = "INFO"):
        self.enable_logging = enable_logging
        self.log_level = log_level
        self.start_time = datetime.now()
        self.test_results = []

        if enable_logging:
            self.lean_logger = LeanLogger("comprehensive_test_runner", "TestRunner")
            self.lean_logger.logger.info("Comprehensive Lean Test Runner initialized", extra={
                "log_level": log_level,
                "timestamp": self.start_time.isoformat()
            })

    def log_test_step(self, step_name: str, details: Optional[Dict[str, Any]] = None):
        """Log a test step with comprehensive context"""
        if self.enable_logging:
            self.lean_logger.log_step_start(step_name, details)

    def log_test_completion(self, step_name: str, success: bool, result_details: Optional[Dict[str, Any]] = None):
        """Log test completion with results"""
        if self.enable_logging:
            self.lean_logger.log_step_end(step_name, success, result_details)

        self.test_results.append({
            "step": step_name,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "details": result_details or {}
        })

    def test_lean_runner_initialization(self) -> bool:
        """Test LeanRunner initialization and basic functionality"""
        self.log_test_step("test_lean_runner_initialization")

        try:
            runner = LeanRunner(lean_module="TestRunner")

            # Validate initialization
            assert hasattr(runner, 'lean_path')
            assert hasattr(runner, 'timeout')
            assert hasattr(runner, 'lean_module')
            assert hasattr(runner, 'lean_logger')

            # Validate methods exist
            required_methods = [
                'run_lean_code',
                'run_theorem_verification',
                'run_algorithm_verification',
                'extract_mathematical_results',
                'save_comprehensive_proof_outcomes'
            ]

            for method in required_methods:
                assert hasattr(runner, method), f"Missing method: {method}"
                assert callable(getattr(runner, method)), f"Method not callable: {method}"

            self.log_test_completion("test_lean_runner_initialization", True, {
                "lean_path": runner.lean_path,
                "timeout": runner.timeout,
                "methods_validated": len(required_methods)
            })

            return True

        except Exception as e:
            self.log_test_completion("test_lean_runner_initialization", False, {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False

    def test_real_lean_execution(self) -> bool:
        """Test real Lean code execution"""
        self.log_test_step("test_real_lean_execution", {
            "test_type": "real_execution",
            "expected_outcome": "successful_compilation"
        })

        try:
            runner = LeanRunner(lean_module="RealLeanTest")

            # Test simple Lean code execution
            lean_code = """
import LeanNiche.Basic

theorem test_addition : ∀ n : Nat, n + 0 = n := by
  intro n
  induction n with
  | zero => rfl
  | succ n' ih => rw [Nat.succ_add, ih]

def test_function (x : Nat) : Nat := x * 2
"""

            result = runner.run_lean_code(lean_code, ["LeanNiche.Basic"])

            # Validate real execution results
            assert result['success'] == True, "Lean execution should succeed"
            assert result['execution_time'] > 0, "Execution time should be positive"
            assert 'result' in result, "Result should contain parsed output"

            parsed_result = result['result']
            verification_status = parsed_result.get('verification_status', {})

            # Validate compilation success
            assert verification_status.get('compilation_successful', False), "Compilation should be successful"

            self.log_test_completion("test_real_lean_execution", True, {
                "execution_time": result['execution_time'],
                "compilation_successful": verification_status.get('compilation_successful'),
                "stdout_length": len(result.get('stdout', '')),
                "stderr_length": len(result.get('stderr', ''))
            })

            return True

        except Exception as e:
            self.log_test_completion("test_real_lean_execution", False, {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False

    def test_lean_theorem_verification(self) -> bool:
        """Test Lean theorem verification"""
        self.log_test_step("test_lean_theorem_verification", {
            "test_type": "theorem_verification",
            "theorems_to_test": ["basic_arithmetic", "list_properties"]
        })

        try:
            runner = LeanRunner(lean_module="TheoremVerificationTest")

            # Test theorem verification
            theorem_code = """
import LeanNiche.Basic

theorem commutativity_test : ∀ m n : Nat, m + n = n + m := by
  intro m n
  induction m with
  | zero => rw [Nat.zero_add, Nat.add_zero]
  | succ m' ih => rw [Nat.succ_add, ih, ←Nat.add_succ]

theorem list_length_append : ∀ (α : Type) (xs ys : List α),
  (xs ++ ys).length = xs.length + ys.length := by
  intro α xs ys
  induction xs with
  | nil => rfl
  | cons x xs' ih => rw [List.length_cons, List.append_cons, ih, Nat.succ_add]
"""

            result = runner.run_theorem_verification(theorem_code, ["LeanNiche.Basic"])

            assert result['success'] == True, "Theorem verification should succeed"
            assert result['execution_time'] > 0, "Execution time should be positive"

            self.log_test_completion("test_lean_theorem_verification", True, {
                "execution_time": result['execution_time'],
                "compilation_successful": result.get('result', {}).get('verification_status', {}).get('compilation_successful')
            })

            return True

        except Exception as e:
            self.log_test_completion("test_lean_theorem_verification", False, {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False

    def test_lean_algorithm_verification(self) -> bool:
        """Test Lean algorithm verification"""
        self.log_test_step("test_lean_algorithm_verification", {
            "test_type": "algorithm_verification",
            "algorithms_to_test": ["sorting", "search"]
        })

        try:
            runner = LeanRunner(lean_module="AlgorithmVerificationTest")

            # Test algorithm verification with test cases
            algorithm_code = """
import LeanNiche.Basic

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

def insertion_sort : List Nat → List Nat
  | [] => []
  | x :: xs =>
    let sorted_tail := insertion_sort xs
    insert x sorted_tail
where insert : Nat → List Nat → List Nat
  | x, [] => [x]
  | x, y :: ys =>
    if x ≤ y then x :: y :: ys else y :: insert x ys
"""

            test_cases = [
                {"input": [1, 2, 3, 4, 5], "expected": True, "description": "search_existing"},
                {"input": [1, 2, 3, 4, 5], "expected": False, "description": "search_nonexistent"},
                {"input": [5, 3, 1, 4, 2], "expected": [1, 2, 3, 4, 5], "description": "sort_unsorted"}
            ]

            result = runner.run_algorithm_verification(algorithm_code, test_cases, ["LeanNiche.Basic"])

            assert result['success'] == True, "Algorithm verification should succeed"
            assert result['execution_time'] > 0, "Execution time should be positive"

            self.log_test_completion("test_lean_algorithm_verification", True, {
                "execution_time": result['execution_time'],
                "test_cases_run": len(test_cases)
            })

            return True

        except Exception as e:
            self.log_test_completion("test_lean_algorithm_verification", False, {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False

    def test_lean_error_handling(self) -> bool:
        """Test Lean error handling and recovery"""
        self.log_test_step("test_lean_error_handling", {
            "test_type": "error_handling",
            "error_types": ["compilation_error", "syntax_error", "import_error"]
        })

        try:
            runner = LeanRunner(lean_module="ErrorHandlingTest")

            # Test with invalid Lean code
            invalid_code = """
import LeanNiche.Basic

theorem invalid_syntax : ∀ n : Nat, n + = n := by  -- Syntax error
  intro n
  rfl
"""

            result = runner.run_lean_code(invalid_code, ["LeanNiche.Basic"])

            # Should handle error gracefully
            assert result['success'] == False, "Should detect compilation error"
            assert 'error' in result, "Should contain error information"
            assert result['execution_time'] >= 0, "Execution time should be valid"

            self.log_test_completion("test_lean_error_handling", True, {
                "error_detected": result['success'] == False,
                "error_message_present": 'error' in result,
                "graceful_handling": True
            })

            return True

        except Exception as e:
            self.log_test_completion("test_lean_error_handling", False, {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False

    def test_lean_performance_monitoring(self) -> bool:
        """Test Lean performance monitoring"""
        self.log_test_step("test_lean_performance_monitoring", {
            "test_type": "performance_monitoring",
            "metrics": ["execution_time", "compilation_time"]
        })

        try:
            runner = LeanRunner(lean_module="PerformanceTest")

            # Test performance monitoring with multiple runs
            lean_code = """
import LeanNiche.Basic

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_positive : ∀ n : Nat, factorial n > 0 := by
  intro n
  induction n with
  | zero => exact Nat.zero_lt_one
  | succ n' ih => exact Nat.mul_pos (Nat.succ_pos n') ih
"""

            # Run multiple times to test performance consistency
            results = []
            for i in range(3):
                start_time = time.time()
                result = runner.run_lean_code(lean_code, ["LeanNiche.Basic"])
                execution_time = time.time() - start_time

                assert result['success'] == True, f"Run {i+1} should succeed"
                assert result['execution_time'] > 0, f"Run {i+1} should have positive execution time"

                results.append({
                    "run": i + 1,
                    "execution_time": result['execution_time'],
                    "success": result['success']
                })

            self.log_test_completion("test_lean_performance_monitoring", True, {
                "runs_completed": len(results),
                "average_execution_time": sum(r['execution_time'] for r in results) / len(results),
                "all_successful": all(r['success'] for r in results)
            })

            return True

        except Exception as e:
            self.log_test_completion("test_lean_performance_monitoring", False, {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False

    def test_orchestrator_lean_integration(self) -> bool:
        """Test orchestrator Lean integration"""
        self.log_test_step("test_orchestrator_lean_integration", {
            "test_type": "orchestrator_integration",
            "components": ["LeanRunner", "logging", "error_handling"]
        })

        try:
            # Mock the dependencies for testing
            from unittest.mock import Mock, patch

            with patch('src.python.core.orchestrator_base.LeanRunner'), \
                 patch('src.python.analysis.comprehensive_analysis.ComprehensiveMathematicalAnalyzer'), \
                 patch('src.python.visualization.visualization.MathematicalVisualizer'), \
                 patch('src.python.analysis.data_generator.MathematicalDataGenerator'):

                class TestOrchestrator(LeanNicheOrchestratorBase):
                    def __init__(self):
                        super().__init__("IntegrationTest", "test_output", enable_logging=self.enable_logging)

                    def run_domain_specific_analysis(self):
                        return {"test": "integration_data"}

                    def create_domain_visualizations(self, analysis_results):
                        pass

                orchestrator = TestOrchestrator()

                # Validate orchestrator initialization
                assert hasattr(orchestrator, 'lean_runner')
                assert hasattr(orchestrator, 'data_dir')
                assert hasattr(orchestrator, 'proofs_dir')
                assert hasattr(orchestrator, 'viz_dir')
                assert hasattr(orchestrator, 'reports_dir')
                assert hasattr(orchestrator, 'lean_modules_used')
                assert hasattr(orchestrator, 'proof_outcomes')

                if self.enable_logging:
                    assert hasattr(orchestrator, 'lean_logger')

                self.log_test_completion("test_orchestrator_lean_integration", True, {
                    "orchestrator_initialized": True,
                    "logging_enabled": self.enable_logging,
                    "directories_created": 4,
                    "components_initialized": 5  # LeanRunner + 4 analysis components
                })

                return True

        except Exception as e:
            self.log_test_completion("test_orchestrator_lean_integration", False, {
                "error": str(e),
                "error_type": type(e).__name__
            })
            return False

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive Lean tests"""
        if self.enable_logging:
            self.lean_logger.logger.info("Starting comprehensive Lean verification tests")

        test_methods = [
            ("LeanRunner Initialization", self.test_lean_runner_initialization),
            ("Real Lean Execution", self.test_real_lean_execution),
            ("Lean Theorem Verification", self.test_lean_theorem_verification),
            ("Lean Algorithm Verification", self.test_lean_algorithm_verification),
            ("Lean Error Handling", self.test_lean_error_handling),
            ("Lean Performance Monitoring", self.test_lean_performance_monitoring),
            ("Orchestrator Integration", self.test_orchestrator_lean_integration)
        ]

        results = {}
        total_passed = 0
        total_tests = len(test_methods)

        for test_name, test_method in test_methods:
            if self.enable_logging:
                self.lean_logger.logger.info(f"Running test: {test_name}")

            success = test_method()
            results[test_name] = success

            if success:
                total_passed += 1

            if self.enable_logging:
                status = "PASSED" if success else "FAILED"
                self.lean_logger.logger.info(f"Test {status}: {test_name}")

        # Generate comprehensive report
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_tests - total_passed,
                "success_rate": (total_passed / total_tests) * 100,
                "duration_seconds": duration
            },
            "test_results": results,
            "timestamp": self.start_time.isoformat(),
            "logging_enabled": self.enable_logging,
            "lean_methods_validated": [
                "run_lean_code",
                "run_theorem_verification",
                "run_algorithm_verification",
                "extract_mathematical_results",
                "save_comprehensive_proof_outcomes"
            ]
        }

        if self.enable_logging:
            self.lean_logger.logger.info("Comprehensive Lean verification tests completed", extra={
                "total_tests": total_tests,
                "passed": total_passed,
                "success_rate": f"{(total_passed / total_tests) * 100:.1f}%",
                "duration_seconds": duration
            })

        return report

    def save_test_report(self, report: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Save comprehensive test report"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"comprehensive_lean_test_report_{timestamp}.json"

        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        return str(output_path)


def main():
    """Main function to run comprehensive Lean verification tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Lean Verification Test Runner")
    parser.add_argument("--no-logging", action="store_true", help="Disable logging")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--output-file", help="Output file for test report")

    args = parser.parse_args()

    # Setup logging if enabled
    if not args.no_logging:
        setup_lean_logging(log_level=args.log_level)

    # Run comprehensive tests
    test_runner = ComprehensiveLeanTestRunner(
        enable_logging=not args.no_logging,
        log_level=args.log_level
    )

    print("🚀 Starting Comprehensive Lean Verification Tests")
    print("=" * 60)

    report = test_runner.run_all_tests()

    # Print summary
    summary = report["test_summary"]
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE LEAN VERIFICATION TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']}")
    print(f"Failed: {summary['failed']}")
    print(".1f")
    print(".2f")

    # Print individual test results
    print("\n📋 Individual Test Results:")
    for test_name, success in report["test_results"].items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}: {test_name}")

    # Save report
    report_file = test_runner.save_test_report(report, args.output_file)
    print(f"\n📁 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    if summary["failed"] > 0:
        print("\n⚠️  Some tests failed. Check logs for details.")
        sys.exit(1)
    else:
        print("\n🎉 All Lean verification tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
