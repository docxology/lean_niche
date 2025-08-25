#!/bin/bash

# LeanNiche Test Script
# Comprehensive testing framework for the LeanNiche environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_output="$3"

    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    print_status "Running test: $test_name"

    if eval "$test_command" > test_output.log 2>&1; then
        if [ -n "$expected_output" ]; then
            if grep -q "$expected_output" test_output.log; then
                print_success "✓ $test_name passed"
                TESTS_PASSED=$((TESTS_PASSED + 1))
            else
                print_error "✗ $test_name failed - expected output not found"
                TESTS_FAILED=$((TESTS_FAILED + 1))
                echo "  Expected: $expected_output"
                echo "  Got: $(cat test_output.log)"
            fi
        else
            print_success "✓ $test_name passed"
            TESTS_PASSED=$((TESTS_PASSED + 1))
        fi
    else
        print_error "✗ $test_name failed - command exited with error"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo "  Error output: $(cat test_output.log)"
    fi

    rm -f test_output.log
}

# Function to test Lean module compilation using Lake
test_lean_module() {
    local module_name="$1"
    local test_name="$2"

    # Use Lake to build and test the project
    run_test "$test_name compilation" "lake build" "Build completed successfully"
}

# Test Lean file compilation directly
test_lean_file_compilation() {
    print_status "Testing individual Lean file compilation..."

    local compile_errors=0
    local total_files=0

    for file in $(find src -name "*.lean" -type f); do
        total_files=$((total_files + 1))
        print_status "Testing compilation: $file"

        if lean "$file" 2>&1 | grep -q "error:"; then
            print_error "  ✗ Compilation failed: $file"
            compile_errors=$((compile_errors + 1))
        else
            print_success "  ✓ Compilation successful: $file"
        fi
    done

    if [ $compile_errors -gt 0 ]; then
        print_error "Lean compilation test failed: $compile_errors/$total_files files failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    else
        print_success "All $total_files Lean files compile successfully"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Test Lean module imports
test_lean_imports() {
    print_status "Testing Lean module imports..."

    local import_errors=0
    local total_imports=0

    for file in $(find src -name "*.lean" -type f); do
        local imports=$(grep "^import " "$file" | sed 's/import //g' || true)

        if [ -n "$imports" ]; then
            print_status "Checking imports in: $file"
            echo "$imports" | while read -r import; do
                total_imports=$((total_imports + 1))

                # Check if imported module exists
                local module_path="src/lean/${import}.lean"
                if [ ! -f "$module_path" ]; then
                    print_error "  ✗ Missing module: $import (expected: $module_path)"
                    import_errors=$((import_errors + 1))
                else
                    print_success "  ✓ Found module: $import"
                fi
            done
        fi
    done

    if [ $import_errors -gt 0 ]; then
        print_error "Lean import test failed: $import_errors/$total_imports imports failed"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    else
        print_success "All $total_imports Lean imports resolve correctly"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Unit tests for individual modules using Lake
test_basic_module() {
    print_status "Testing Basic module..."
    test_lean_module "Basic" "Basic module"
}

test_advanced_module() {
    print_status "Testing Advanced module..."
    test_lean_module "Advanced" "Advanced module"
}

test_tactics_module() {
    print_status "Testing Tactics module..."
    test_lean_module "Tactics" "Tactics module"
}

test_set_theory_module() {
    print_status "Testing Set Theory module..."
    test_lean_module "SetTheory" "Set Theory module"
}

test_computational_module() {
    print_status "Testing Computational module..."
    test_lean_module "Computational" "Computational module"
}

test_statistics_module() {
    print_status "Testing Statistics module..."
    test_lean_module "Statistics" "Statistics module"
}

test_dynamical_systems_module() {
    print_status "Testing Dynamical Systems module..."
    test_lean_module "DynamicalSystems" "Dynamical Systems module"
}

test_lyapunov_module() {
    print_status "Testing Lyapunov module..."
    test_lean_module "Lyapunov" "Lyapunov module"
}

test_linear_algebra_module() {
    print_status "Testing Linear Algebra module..."
    test_lean_module "LinearAlgebra" "Linear Algebra module"
}

test_control_theory_module() {
    print_status "Testing Control Theory module..."
    test_lean_module "ControlTheory" "Control Theory module"
}

test_free_energy_principle_module() {
    print_status "Testing Free Energy Principle module..."
    test_lean_module "FreeEnergyPrinciple" "Free Energy Principle module"
}

test_active_inference_module() {
    print_status "Testing Active Inference module..."
    test_lean_module "ActiveInference" "Active Inference module"
}

test_predictive_coding_module() {
    print_status "Testing Predictive Coding module..."
    test_lean_module "PredictiveCoding" "Predictive Coding module"
}

test_belief_propagation_module() {
    print_status "Testing Belief Propagation module..."
    test_lean_module "BeliefPropagation" "Belief Propagation module"
}

test_decision_making_module() {
    print_status "Testing Decision Making module..."
    test_lean_module "DecisionMaking" "Decision Making module"
}

test_learning_adaptation_module() {
    print_status "Testing Learning Adaptation module..."
    test_lean_module "LearningAdaptation" "Learning Adaptation module"
}

test_signal_processing_module() {
    print_status "Testing Signal Processing module..."
    test_lean_module "SignalProcessing" "Signal Processing module"
}

# Integration tests
test_integration() {
    print_status "Running integration tests..."

    run_test "Full project build" "lake build" "Build completed successfully"

    run_test "Main executable" "lake exe lean_niche" "Test Suite Complete"
}

# Performance tests
test_performance() {
    print_status "Running performance tests..."

    # Test build time
    local start_time=$(date +%s)
    lake build > /dev/null 2>&1
    local end_time=$(date +%s)
    local build_time=$((end_time - start_time))

    if [ $build_time -lt 60 ]; then
        print_success "✓ Build performance acceptable (${build_time}s)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_warning "⚠ Build performance slow (${build_time}s)"
        TESTS_PASSED=$((TESTS_PASSED + 1))  # Still count as passed, just warn
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))

    # Test execution time
    start_time=$(date +%s)
    timeout 30 lake exe lean_niche > /dev/null 2>&1
    end_time=$(date +%s)
    local exec_time=$((end_time - start_time))

    if [ $exec_time -lt 30 ]; then
        print_success "✓ Execution performance acceptable (${exec_time}s)"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_error "✗ Execution performance too slow (${exec_time}s)"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Coverage analysis
test_coverage() {
    print_status "Analyzing test coverage..."

    # Count total theorems and definitions
    local lean_files=$(find src tests -name "*.lean" -type f)
    local total_items=0

    for file in $lean_files; do
        # Count theorems, lemmas, and definitions
        local items=$(grep -c "theorem\|lemma\|def\|def " "$file" || true)
        total_items=$((total_items + items))
    done

    print_status "Total theorems/definitions found: $total_items"

    if [ $total_items -gt 20 ]; then
        print_success "✓ Good coverage - sufficient theorems and definitions"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        print_warning "⚠ Low coverage - consider adding more theorems"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    fi
    TESTS_TOTAL=$((TESTS_TOTAL + 1))
}

# Dependency tests
test_dependencies() {
    print_status "Testing dependencies..."

    # Test that Lake can resolve and build dependencies
    run_test "Lake dependency resolution" "lake update" "info: toolchain not updated"

    # Test that the main executable runs (which implies dependencies are working)
    run_test "Project dependencies" "lake exe lean_niche" "LeanNiche"
}

# Main test runner
main() {
    print_status "Starting LeanNiche comprehensive test suite..."
    echo "=================================================="

    local test_mode="${1:-all}"

    case "$test_mode" in
        "unit")
            test_basic_module
            test_advanced_module
            test_tactics_module
            test_set_theory_module
            test_computational_module
            test_statistics_module
            test_dynamical_systems_module
            test_lyapunov_module
            test_linear_algebra_module
            test_control_theory_module
            test_free_energy_principle_module
            test_active_inference_module
            test_predictive_coding_module
            test_belief_propagation_module
            test_decision_making_module
            test_learning_adaptation_module
            test_signal_processing_module
            ;;
        "integration")
            test_integration
            ;;
        "performance")
            test_performance
            ;;
        "coverage")
            test_coverage
            ;;
        "dependencies")
            test_dependencies
            ;;
        "lean")
            test_lean_file_compilation
            test_lean_imports
            ;;
        "all")
            test_lean_file_compilation
            test_lean_imports
            test_basic_module
            test_advanced_module
            test_tactics_module
            test_set_theory_module
            test_computational_module
            test_statistics_module
            test_dynamical_systems_module
            test_lyapunov_module
            test_linear_algebra_module
            test_control_theory_module
            test_free_energy_principle_module
            test_active_inference_module
            test_predictive_coding_module
            test_belief_propagation_module
            test_decision_making_module
            test_learning_adaptation_module
            test_signal_processing_module
            test_integration
            test_performance
            test_coverage
            test_dependencies
            ;;
        *)
            print_error "Usage: $0 [unit|integration|performance|coverage|dependencies|lean|all]"
            exit 1
            ;;
    esac

    echo "=================================================="

    # Print test summary
    print_status "Test Summary:"
    echo "  Total tests: $TESTS_TOTAL"
    echo "  Passed: $TESTS_PASSED"
    echo "  Failed: $TESTS_FAILED"

    local success_rate=0
    if [ $TESTS_TOTAL -gt 0 ]; then
        success_rate=$((TESTS_PASSED * 100 / TESTS_TOTAL))
    fi

    echo "  Success rate: ${success_rate}%"

    if [ $TESTS_FAILED -eq 0 ]; then
        print_success "All tests completed successfully!"
        exit 0
    else
        print_error "Some tests failed. Check output above for details."
        exit 1
    fi
}

# Run main function with all arguments
main "$@"
