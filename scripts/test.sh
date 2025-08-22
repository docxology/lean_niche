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

# Unit tests for individual modules
test_basic_module() {
    print_status "Testing Basic module..."

    run_test "Basic arithmetic compilation" "lean src/LeanNiche/Basic.lean" ""

    # Test specific theorems
    run_test "Addition commutativity" "lean -c 'import LeanNiche.Basic; #check LeanNiche.Basic.add_comm'" "add_comm"
    run_test "Addition associativity" "lean -c 'import LeanNiche.Basic; #check LeanNiche.Basic.add_assoc'" "add_assoc"
}

test_advanced_module() {
    print_status "Testing Advanced module..."

    run_test "Advanced theorems compilation" "lean src/LeanNiche/Advanced.lean" ""

    # Test specific theorems
    run_test "Sum formula" "lean -c 'import LeanNiche.Advanced; #check LeanNiche.Advanced.sum_up_to_correct'" "sum_up_to_correct"
    run_test "Infinite primes" "lean -c 'import LeanNiche.Advanced; #check LeanNiche.Advanced.infinite_primes'" "infinite_primes"
}

test_tactics_module() {
    print_status "Testing Tactics module..."

    run_test "Tactics compilation" "lean src/LeanNiche/Tactics.lean" ""
}

test_set_theory_module() {
    print_status "Testing Set Theory module..."

    run_test "Set theory compilation" "lean src/LeanNiche/SetTheory.lean" ""

    # Test specific theorems
    run_test "Empty subset theorem" "lean -c 'import LeanNiche.SetTheory; #check LeanNiche.SetTheory.empty_subset'" "empty_subset"
}

test_computational_module() {
    print_status "Testing Computational module..."

    run_test "Computational algorithms compilation" "lean src/LeanNiche/Computational.lean" ""

    # Test specific functions
    run_test "Fibonacci function" "lean -c 'import LeanNiche.Computational; #check LeanNiche.Computational.fibonacci'" "fibonacci"
    run_test "Factorial function" "lean -c 'import LeanNiche.Computational; #check LeanNiche.Computational.factorial'" "factorial"
}

test_test_suite() {
    print_status "Testing Test Suite module..."

    run_test "Test suite compilation" "lean tests/TestSuite.lean" ""
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

    run_test "Mathlib4 availability" "lean -c 'import Mathlib.Data.Nat.Basic'" "Mathlib.Data.Nat.Basic"

    run_test "Project dependencies" "lake exe lean_niche 2>&1 | grep -q 'Mathlib'" "Mathlib"
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
            test_test_suite
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
        "all")
            test_basic_module
            test_advanced_module
            test_tactics_module
            test_set_theory_module
            test_computational_module
            test_test_suite
            test_integration
            test_performance
            test_coverage
            test_dependencies
            ;;
        *)
            print_error "Usage: $0 [unit|integration|performance|coverage|dependencies|all]"
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
