#!/usr/bin/env python3
"""
LeanNiche Comprehensive Test Runner

This script runs all tests for the LeanNiche environment including:
- Python module tests (using pytest)
- Lean module tests (using Lake/Lean)
- LaTeX conversion tests (using pytest)
- Integration tests
- Performance tests

Usage:
    python run_tests.py [options]

Options:
    --python-only        Run only Python tests
    --lean-only         Run only Lean tests
    --latex-only        Run only LaTeX tests
    --integration-only  Run only integration tests
    --performance-only  Run only performance tests
    --coverage          Generate coverage report
    --verbose           Verbose output
    --parallel          Run tests in parallel (Python only)
    --fail-fast         Stop on first failure
    --junit-xml FILE    Generate JUnit XML report
"""

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
MAGENTA = '\033[0;35m'
CYAN = '\033[0;36m'
NC = '\033[0m'  # No Color


class TestRunner:
    """Main test runner class"""

    def __init__(self, verbose: bool = False, fail_fast: bool = False):
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.test_results = {}
        self.start_time = None
        self.project_root = Path(__file__).parent.parent.parent

    def print_status(self, message: str, color: str = BLUE):
        """Print colored status message"""
        print(f"{color}[INFO]{NC} {message}")

    def print_success(self, message: str):
        """Print success message"""
        print(f"{GREEN}[SUCCESS]{NC} {message}")

    def print_error(self, message: str):
        """Print error message"""
        print(f"{RED}[ERROR]{NC} {message}")

    def print_warning(self, message: str):
        """Print warning message"""
        print(f"{YELLOW}[WARNING]{NC} {message}")

    def print_header(self, message: str):
        """Print section header"""
        print(f"\n{MAGENTA}{'=' * 60}{NC}")
        print(f"{MAGENTA}  {message}{NC}")
        print(f"{MAGENTA}{'=' * 60}{NC}")

    def run_command(self, cmd: List[str], cwd: Optional[Path] = None,
                   env: Optional[Dict] = None, timeout: int = 300) -> Tuple[int, str, str]:
        """Run a command and return (return_code, stdout, stderr)"""
        if self.verbose:
            self.print_status(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            self.print_error(f"Command timed out after {timeout} seconds")
            return -1, "", "Timeout expired"
        except Exception as e:
            self.print_error(f"Command failed: {e}")
            return -1, "", str(e)

    def run_python_tests(self, coverage: bool = False, parallel: bool = False,
                        junit_xml: Optional[str] = None) -> bool:
        """Run Python tests using pytest"""
        self.print_header("Python Module Tests")

        cmd = ["python", "-m", "pytest"]

        if coverage:
            cmd.extend(["--cov=src.python", "--cov-report=term-missing",
                       "--cov-report=html:htmlcov"])

        if parallel:
            cmd.extend(["-n", "auto"])

        if junit_xml:
            cmd.extend(["--junit-xml", junit_xml])

        cmd.extend([
            "src/tests/python/",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--no-header"
        ])

        if self.fail_fast:
            cmd.append("-x")

        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            self.print_success("Python tests passed")
            if self.verbose and stdout:
                print(stdout)
            return True
        else:
            self.print_error("Python tests failed")
            if stderr:
                print(f"{RED}Error output:{NC}")
                print(stderr)
            if self.verbose and stdout:
                print(f"{YELLOW}Standard output:{NC}")
                print(stdout)
            return False

    def run_lean_tests(self) -> bool:
        """Run Lean tests using Lake"""
        self.print_header("Lean Module Tests")

        # Check if Lake is available
        returncode, stdout, stderr = self.run_command(["which", "lake"])
        if returncode != 0:
            self.print_error("Lake not found. Please install Lean 4 and ensure it's in PATH")
            return False

        # Run Lean tests
        returncode, stdout, stderr = self.run_command(["lake", "exe", "lean_niche"])

        if returncode == 0:
            self.print_success("Lean tests passed")
            if self.verbose and stdout:
                print(stdout)
            return True
        else:
            self.print_error("Lean tests failed")
            if stderr:
                print(f"{RED}Error output:{NC}")
                print(stderr)
            if self.verbose and stdout:
                print(f"{YELLOW}Standard output:{NC}")
                print(stdout)
            return False

    def run_latex_tests(self, coverage: bool = False, junit_xml: Optional[str] = None) -> bool:
        """Run LaTeX conversion tests"""
        self.print_header("LaTeX Conversion Tests")

        cmd = ["python", "-m", "pytest"]

        if coverage:
            cmd.extend(["--cov=src.latex", "--cov-report=term-missing"])

        if junit_xml:
            cmd.extend(["--junit-xml", junit_xml])

        cmd.extend([
            "src/tests/latex/",
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--no-header"
        ])

        if self.fail_fast:
            cmd.append("-x")

        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            self.print_success("LaTeX tests passed")
            if self.verbose and stdout:
                print(stdout)
            return True
        else:
            self.print_error("LaTeX tests failed")
            if stderr:
                print(f"{RED}Error output:{NC}")
                print(stderr)
            return False

    def run_integration_tests(self) -> bool:
        """Run integration tests"""
        self.print_header("Integration Tests")

        # Test that all modules can be imported together
        cmd = [
            "python", "-c",
            """
import sys
sys.path.insert(0, 'src')

try:
    # Test Python imports
    from python import cli, visualization
    print('✅ Python modules imported successfully')

    # Test LaTeX imports
    from latex import lean_to_latex
    print('✅ LaTeX modules imported successfully')

    print('✅ All integration tests passed')
except Exception as e:
    print(f'❌ Integration test failed: {e}')
    sys.exit(1)
"""
        ]

        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            self.print_success("Integration tests passed")
            if stdout:
                print(stdout)
            return True
        else:
            self.print_error("Integration tests failed")
            if stderr:
                print(f"{RED}Error output:{NC}")
                print(stderr)
            return False

    def run_performance_tests(self) -> bool:
        """Run performance tests"""
        self.print_header("Performance Tests")

        # Test Python performance
        python_perf_cmd = [
            "python", "-c",
            """
import sys
import time
sys.path.insert(0, 'src')
from python.visualization import MathematicalVisualizer

# Test visualization performance
start_time = time.time()
viz = MathematicalVisualizer()
for i in range(10):
    fig = viz.plot_function(lambda x: x**2, (-5, 5), f'Test Plot {i}')
end_time = time.time()

print(f'✅ Python visualization performance: {end_time - start_time:.3f}s for 10 plots')
"""
        ]

        returncode1, stdout1, stderr1 = self.run_command(python_perf_cmd)

        # Test LaTeX conversion performance
        latex_perf_cmd = [
            "python", "-c",
            """
import sys
import time
sys.path.insert(0, 'src')
from latex.lean_to_latex import LeanToLatexConverter

# Test LaTeX conversion performance
start_time = time.time()
converter = LeanToLatexConverter()

# Test symbol conversion
for i in range(1000):
    result = converter.convert_expression('∀ x ∈ ℝ, x ≥ 0')

end_time = time.time()
print(f'✅ LaTeX conversion performance: {end_time - start_time:.3f}s for 1000 conversions')
"""
        ]

        returncode2, stdout2, stderr2 = self.run_command(latex_perf_cmd)

        success = returncode1 == 0 and returncode2 == 0

        if success:
            self.print_success("Performance tests passed")
            if stdout1:
                print(stdout1)
            if stdout2:
                print(stdout2)
        else:
            self.print_error("Performance tests failed")
            if stderr1:
                print(f"{RED}Python perf error:{NC}")
                print(stderr1)
            if stderr2:
                print(f"{RED}LaTeX perf error:{NC}")
                print(stderr2)

        return success

    def generate_coverage_report(self) -> bool:
        """Generate comprehensive coverage report"""
        self.print_header("Coverage Analysis")

        # Run tests with coverage
        cmd = [
            "python", "-m", "pytest",
            "--cov=src.python",
            "--cov=src.latex",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "src/tests/python/",
            "src/tests/latex/",
            "-q"
        ]

        returncode, stdout, stderr = self.run_command(cmd)

        if returncode == 0:
            self.print_success("Coverage report generated")
            if stdout:
                print(stdout)
            return True
        else:
            self.print_error("Coverage report generation failed")
            if stderr:
                print(f"{RED}Error output:{NC}")
                print(stderr)
            return False

    def run_all_tests(self, test_types: List[str] = None,
                     coverage: bool = False, parallel: bool = False,
                     junit_xml: Optional[str] = None) -> bool:
        """Run all tests"""
        self.start_time = time.time()

        if test_types is None:
            test_types = ["python", "lean", "latex", "integration", "performance"]

        results = {}

        # Run tests in specified order
        test_functions = {
            "python": lambda: self.run_python_tests(coverage, parallel, junit_xml),
            "lean": self.run_lean_tests,
            "latex": lambda: self.run_latex_tests(coverage, junit_xml),
            "integration": self.run_integration_tests,
            "performance": self.run_performance_tests
        }

        all_passed = True

        for test_type in test_types:
            if test_type in test_functions:
                try:
                    result = test_functions[test_type]()
                    results[test_type] = result
                    if not result and self.fail_fast:
                        self.print_error(f"Stopping due to {test_type} test failure")
                        all_passed = False
                        break
                    all_passed = all_passed and result
                except Exception as e:
                    self.print_error(f"Error running {test_type} tests: {e}")
                    results[test_type] = False
                    all_passed = False
                    if self.fail_fast:
                        break

        # Generate coverage report if requested and not already done
        if coverage and "python" in test_types and "latex" in test_types:
            self.generate_coverage_report()

        # Print summary
        self.print_test_summary(results, time.time() - self.start_time)

        return all_passed

    def print_test_summary(self, results: Dict[str, bool], duration: float):
        """Print test summary"""
        self.print_header("Test Summary")

        print(f"{CYAN}Total test time: {duration:.2f} seconds{NC}")
        print()

        passed = sum(1 for result in results.values() if result)
        total = len(results)

        print(f"Tests completed: {total}")
        print(f"Tests passed: {passed}")
        print(f"Tests failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        print()

        for test_type, result in results.items():
            status = f"{GREEN}✅ PASSED{NC}" if result else f"{RED}❌ FAILED{NC}"
            print(f"  {test_type.capitalize():12} {status}")

        if all(results.values()):
            self.print_success("🎉 All tests completed successfully!")
        else:
            self.print_error("⚠️  Some tests failed. Check output above for details.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="LeanNiche Test Runner")
    parser.add_argument("--python-only", action="store_true",
                       help="Run only Python tests")
    parser.add_argument("--lean-only", action="store_true",
                       help="Run only Lean tests")
    parser.add_argument("--latex-only", action="store_true",
                       help="Run only LaTeX tests")
    parser.add_argument("--integration-only", action="store_true",
                       help="Run only integration tests")
    parser.add_argument("--performance-only", action="store_true",
                       help="Run only performance tests")
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--parallel", action="store_true",
                       help="Run Python tests in parallel")
    parser.add_argument("--fail-fast", "-x", action="store_true",
                       help="Stop on first failure")
    parser.add_argument("--junit-xml",
                       help="Generate JUnit XML report")

    args = parser.parse_args()

    # Determine which tests to run
    if args.python_only:
        test_types = ["python"]
    elif args.lean_only:
        test_types = ["lean"]
    elif args.latex_only:
        test_types = ["latex"]
    elif args.integration_only:
        test_types = ["integration"]
    elif args.performance_only:
        test_types = ["performance"]
    else:
        test_types = ["python", "lean", "latex", "integration", "performance"]

    # Create test runner
    runner = TestRunner(verbose=args.verbose, fail_fast=args.fail_fast)

    # Print header
    runner.print_header("LeanNiche Comprehensive Test Suite")
    print(f"{CYAN}Testing modules: {', '.join(test_types)}{NC}")
    if args.coverage:
        print(f"{CYAN}Coverage report will be generated{NC}")
    print()

    # Run tests
    success = runner.run_all_tests(
        test_types=test_types,
        coverage=args.coverage,
        parallel=args.parallel,
        junit_xml=args.junit_xml
    )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
