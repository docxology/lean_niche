#!/bin/bash

# LeanNiche Build Script
# Comprehensive build and verification script for the LeanNiche environment

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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."

    if ! command_exists lake; then
        print_error "Lake (Lean package manager) not found"
        print_status "Please install Elan: https://leanprover-community.github.io/get_started.html"
        exit 1
    fi

    if ! command_exists lean; then
        print_error "Lean not found"
        print_status "Please ensure Lean is properly installed via Elan"
        exit 1
    fi

    print_success "Prerequisites check passed"
}

# Clean build artifacts
clean_build() {
    print_status "Cleaning build artifacts..."
    lake clean
    rm -rf build/
    print_success "Clean completed"
}

# Update dependencies
update_dependencies() {
    print_status "Updating dependencies..."
    lake update
    print_success "Dependencies updated"
}

# Build the project
build_project() {
    print_status "Building LeanNiche project..."

    if lake build; then
        print_success "Project built successfully"
    else
        print_error "Build failed"
        exit 1
    fi
}

# Verify Lean module structure
verify_lean_modules() {
    print_status "Verifying Lean module structure..."

    # Check for core LeanNiche modules
    local required_modules=(
        "src/lean/LeanNiche/Basic.lean"
        "src/lean/LeanNiche/Advanced.lean"
        "src/lean/LeanNiche/Computational.lean"
        "src/lean/LeanNiche/Main.lean"
    )

    local missing_modules=0
    for module in "${required_modules[@]}"; do
        if [ ! -f "$module" ]; then
            print_error "Missing required module: $module"
            missing_modules=$((missing_modules + 1))
        else
            print_success "Found: $module"
        fi
    done

    if [ $missing_modules -gt 0 ]; then
        print_error "Missing $missing_modules required Lean modules"
        return 1
    else
        print_success "All core Lean modules present"
        return 0
    fi
}

# Test Lean compilation
test_lean_compilation() {
    print_status "Testing Lean compilation..."

    local compile_errors=0
    local total_files=0

    for file in $(find src -name "*.lean" -type f); do
        total_files=$((total_files + 1))
        print_status "Testing: $file"

        if lean "$file" 2>&1 | grep -q "error:"; then
            print_error "  ✗ Compilation failed: $file"
            compile_errors=$((compile_errors + 1))
        else
            print_success "  ✓ Compilation successful: $file"
        fi
    done

    if [ $compile_errors -gt 0 ]; then
        print_error "Lean compilation test failed: $compile_errors/$total_files files failed"
        return 1
    else
        print_success "All $total_files Lean files compile successfully"
        return 0
    fi
}

# Run tests
run_tests() {
    print_status "Running test suite..."
    if lake exe lean_niche; then
        print_success "All tests passed"
    else
        print_error "Some tests failed"
        exit 1
    fi
}

# Generate documentation
generate_docs() {
    print_status "Generating documentation..."
    # This would typically use a documentation generator
    # For now, we'll just check if docs exist
    if [ -d "docs" ]; then
        print_success "Documentation exists"
    else
        print_warning "No documentation directory found"
    fi
}

# Performance analysis
performance_analysis() {
    print_status "Running performance analysis..."

    # Time the build process
    local start_time=$(date +%s)
    lake build > /dev/null 2>&1
    local end_time=$(date +%s)
    local build_time=$((end_time - start_time))

    print_status "Build time: ${build_time}s"

    # Check proof compilation times
    print_status "Checking proof compilation..."
    if timeout 30 lake exe lean_niche > /dev/null 2>&1; then
        print_success "Proof compilation within acceptable time"
    else
        print_warning "Proof compilation took longer than expected"
    fi
}

# Code analysis
analyze_code() {
    print_status "Analyzing code structure..."

    # Count lines of code
    local lean_files=$(find src tests -name "*.lean" -type f)
    local total_lines=0
    local file_count=0

    for file in $lean_files; do
        local lines=$(wc -l < "$file")
        total_lines=$((total_lines + lines))
        file_count=$((file_count + 1))
        print_status "  $file: $lines lines"
    done

    print_status "Total: $file_count files, $total_lines lines"
}

# Verify Mathlib4 integration
verify_mathlib() {
    print_status "Verifying Mathlib4 integration..."

    if lake exe lean_niche 2>&1 | grep -q "Mathlib"; then
        print_success "Mathlib4 integration successful"
    else
        print_warning "Mathlib4 may not be properly integrated"
    fi
}

# Main build process
main() {
    print_status "Starting LeanNiche build process..."
    echo "========================================"

    check_prerequisites

    case "${1:-all}" in
        "clean")
            clean_build
            ;;
        "update")
            update_dependencies
            ;;
        "build")
            build_project
            ;;
        "test")
            run_tests
            ;;
        "docs")
            generate_docs
            ;;
        "analyze")
            analyze_code
            ;;
        "perf")
            performance_analysis
            ;;
        "verify")
            verify_mathlib
            ;;
        "lean")
            verify_lean_modules
            test_lean_compilation
            ;;
        "all")
            clean_build
            update_dependencies
            verify_lean_modules
            test_lean_compilation
            build_project
            run_tests
            generate_docs
            analyze_code
            performance_analysis
            verify_mathlib
            ;;
        *)
            print_error "Usage: $0 [clean|update|build|test|docs|analyze|perf|verify|lean|all]"
            exit 1
            ;;
    esac

    echo "========================================"
    print_success "LeanNiche build process completed successfully!"
    print_status "You can now run: lake exe lean_niche"
}

# Run main function with all arguments
main "$@"
