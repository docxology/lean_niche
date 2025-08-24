#!/usr/bin/env python3
"""
Simple Test Runner for LeanNiche

Bypasses pytest configuration issues and directly tests modules.
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Set matplotlib backend before importing visualization modules
os.environ['MPLBACKEND'] = 'Agg'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

def print_status(message, color_code=34):
    """Print colored status message"""
    print(f"\033[{color_code}m{message}\033[0m")

def print_header(message):
    """Print section header"""
    print(f"\n\033[35m{'=' * 60}\033[0m")
    print(f"\033[35m  {message}\033[0m")
    print(f"\033[35m{'=' * 60}\033[0m")

def test_python_modules():
    """Test Python modules directly"""
    print_header("Python Module Tests")

    test_results = []

    try:
        # Test CLI module
        print_status("Testing CLI module...")
        from python.cli import cli, plot_function, analyze_data, gallery, info
        print_status("✅ CLI module imported successfully", 32)

        # Test visualization module
        print_status("Testing visualization module...")
        from python.visualization import (
            MathematicalVisualizer,
            StatisticalAnalyzer,
            DynamicalSystemsVisualizer,
            create_visualization_gallery
        )

        # Test instantiations
        viz = MathematicalVisualizer()
        print_status("✅ MathematicalVisualizer created", 32)

        analyzer = StatisticalAnalyzer()
        print_status("✅ StatisticalAnalyzer created", 32)

        dyn_viz = DynamicalSystemsVisualizer()
        print_status("✅ DynamicalSystemsVisualizer created", 32)

        # Test basic functionality
        result = viz.plot_function(lambda x: x**2, (-5, 5), "Test Plot")
        print_status("✅ Function plotting works", 32)

        test_results.append(("python", True))

    except Exception as e:
        print_status(f"❌ Python test failed: {e}", 31)
        traceback.print_exc()
        test_results.append(("python", False))

    return test_results

def test_latex_modules():
    """Test LaTeX modules directly"""
    print_header("LaTeX Module Tests")

    test_results = []

    try:
        # Test LaTeX converter
        print_status("Testing LaTeX converter...")
        from latex.lean_to_latex import LeanToLatexConverter

        converter = LeanToLatexConverter()
        print_status("✅ LeanToLatexConverter created", 32)

        # Test basic functionality
        result = converter.convert_symbol('∀')
        assert result == r'\forall'
        print_status("✅ Symbol conversion works", 32)

        expr_result = converter.convert_expression('∀ x ∈ ℝ, x ≥ 0')
        assert r'\forall' in expr_result and r'\in' in expr_result
        print_status("✅ Expression conversion works", 32)

        test_results.append(("latex", True))

    except Exception as e:
        print_status(f"❌ LaTeX test failed: {e}", 31)
        traceback.print_exc()
        test_results.append(("latex", False))

    return test_results

def test_integration():
    """Test module integration"""
    print_header("Integration Tests")

    test_results = []

    try:
        print_status("Testing module integration...")

        # Test all imports work together
        from python import cli, visualization
        from latex import lean_to_latex

        print_status("✅ All modules imported successfully", 32)
        test_results.append(("integration", True))

    except Exception as e:
        print_status(f"❌ Integration test failed: {e}", 31)
        traceback.print_exc()
        test_results.append(("integration", False))

    return test_results

def test_performance():
    """Test performance of key operations"""
    print_header("Performance Tests")

    test_results = []

    try:
        print_status("Testing performance...")

        from python.visualization import MathematicalVisualizer
        from latex.lean_to_latex import LeanToLatexConverter

        # Test visualization performance
        start_time = time.time()
        viz = MathematicalVisualizer()
        for i in range(5):  # Reduced for speed
            fig = viz.plot_function(lambda x: x**2, (-5, 5), f'Test Plot {i}')
        viz_time = time.time() - start_time
        print_status(".3f", 32)

        # Test LaTeX conversion performance
        start_time = time.time()
        converter = LeanToLatexConverter()
        for i in range(500):  # Reduced for speed
            result = converter.convert_expression('∀ x ∈ ℝ, x ≥ 0')
        latex_time = time.time() - start_time
        print_status(".3f", 32)

        test_results.append(("performance", True))

    except Exception as e:
        print_status(f"❌ Performance test failed: {e}", 31)
        traceback.print_exc()
        test_results.append(("performance", False))

    return test_results

def run_lean_tests():
    """Run Lean tests using lake"""
    print_header("Lean Module Tests")

    test_results = []

    try:
        print_status("Running Lean tests...")
        import subprocess

        result = subprocess.run(['lake', 'exe', 'lean_niche'],
                              capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            print_status("✅ Lean tests passed", 32)
            test_results.append(("lean", True))
        else:
            print_status(f"❌ Lean tests failed: {result.stderr}", 31)
            test_results.append(("lean", False))

    except Exception as e:
        print_status(f"❌ Lean test execution failed: {e}", 31)
        test_results.append(("lean", False))

    return test_results

def main():
    """Main test runner"""
    print_header("LeanNiche Comprehensive Test Suite")

    all_results = []

    # Run tests
    all_results.extend(test_python_modules())
    all_results.extend(test_latex_modules())
    all_results.extend(test_integration())
    all_results.extend(test_performance())
    all_results.extend(run_lean_tests())

    # Print summary
    print_header("Test Summary")

    passed = sum(1 for _, result in all_results if result)
    total = len(all_results)
    success_rate = passed / total * 100 if total > 0 else 0

    print(f"Tests completed: {total}")
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {total - passed}")
    print(".1f")

    for test_name, result in all_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        color = 32 if result else 31
        print_status(f"  {test_name.capitalize():12} {status}", color)

    if passed == total:
        print_status("🎉 All tests completed successfully!", 32)
        return 0
    else:
        print_status("⚠️  Some tests failed. Check output above for details.", 31)
        return 1

if __name__ == "__main__":
    sys.exit(main())
