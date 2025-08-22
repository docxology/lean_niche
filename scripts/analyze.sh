#!/bin/bash

# LeanNiche Analysis Script
# Code analysis and metrics for the LeanNiche environment

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

# Function to print section headers
print_header() {
    echo ""
    echo "========================================"
    echo "  $1"
    echo "========================================"
}

# Analyze code structure
analyze_structure() {
    print_header "Code Structure Analysis"

    print_status "Analyzing file organization..."

    # Count files by type
    local lean_files=$(find src tests -name "*.lean" -type f | wc -l)
    local total_lines=$(find src tests -name "*.lean" -type f -exec wc -l {} \; | tail -1 | awk '{print $1}')
    local avg_lines=$((total_lines / lean_files))

    print_status "Lean files: $lean_files"
    print_status "Total lines: $total_lines"
    print_status "Average lines per file: $avg_lines"

    # Analyze directory structure
    print_status "Directory structure:"
    find src tests -type f -name "*.lean" | sort | while read -r file; do
        local lines=$(wc -l < "$file")
        printf "  %-40s %6d lines\n" "$file" "$lines"
    done
}

# Analyze theorems and definitions
analyze_theorems() {
    print_header "Theorem and Definition Analysis"

    print_status "Analyzing theorems, lemmas, and definitions..."

    local total_theorems=0
    local total_lemmas=0
    local total_defs=0

    for file in $(find src tests -name "*.lean" -type f); do
        local theorems=$(grep -c "^theorem " "$file" || true)
        local lemmas=$(grep -c "^lemma " "$file" || true)
        local defs=$(grep -c "^def " "$file" || true)

        if [ $((theorems + lemmas + defs)) -gt 0 ]; then
            print_status "$file:"
            [ $theorems -gt 0 ] && print_status "  Theorems: $theorems"
            [ $lemmas -gt 0 ] && print_status "  Lemmas: $lemmas"
            [ $defs -gt 0 ] && print_status "  Definitions: $defs"
        fi

        total_theorems=$((total_theorems + theorems))
        total_lemmas=$((total_lemmas + lemmas))
        total_defs=$((total_defs + defs))
    done

    print_header "Summary"
    print_status "Total theorems: $total_theorems"
    print_status "Total lemmas: $total_lemmas"
    print_status "Total definitions: $total_defs"
    print_status "Total formal objects: $((total_theorems + total_lemmas + total_defs))"
}

# Analyze dependencies and imports
analyze_dependencies() {
    print_header "Dependency Analysis"

    print_status "Analyzing imports and dependencies..."

    # Find all imports
    local all_imports=$(find src tests -name "*.lean" -type f -exec grep -h "^import " {} \; | sort | uniq -c | sort -nr)

    print_status "Most common imports:"
    echo "$all_imports" | head -10 | while read -r count import; do
        echo "  $count $import"
    done

    # Check for circular dependencies (simplified check)
    print_status "Checking for potential issues..."
    local mathlib_imports=$(grep -r "import Mathlib" src tests | wc -l)
    print_status "Mathlib4 imports: $mathlib_imports"

    if [ $mathlib_imports -gt 0 ]; then
        print_success "Mathlib4 integration detected"
    else
        print_warning "No Mathlib4 imports found"
    fi
}

# Analyze proof complexity
analyze_complexity() {
    print_header "Proof Complexity Analysis"

    print_status "Analyzing proof complexity..."

    # Count tactic usage
    local total_tactics=0
    local tactic_types=$(find src tests -name "*.lean" -type f -exec grep -h "by " {} \; | sed 's/by //g' | tr ' ' '\n' | grep -v '^$' | sort | uniq -c | sort -nr)

    print_status "Most common tactics:"
    echo "$tactic_types" | head -10 | while read -r count tactic; do
        echo "  $count $tactic"
        total_tactics=$((total_tactics + count))
    done

    print_status "Total tactic applications: $total_tactics"

    # Analyze proof lengths
    print_status "Analyzing proof lengths..."
    local long_proofs=$(find src tests -name "*.lean" -type f -exec awk '/by/{flag=1; count=0} flag{count++} /^$/{if(flag && count>10){print FILENAME":"NR":"count" lines"; flag=0}}' {} \;)

    if [ -n "$long_proofs" ]; then
        print_warning "Long proofs detected (>10 lines):"
        echo "$long_proofs"
    else
        print_success "No excessively long proofs found"
    fi
}

# Analyze documentation
analyze_documentation() {
    print_header "Documentation Analysis"

    print_status "Analyzing documentation coverage..."

    local documented=0
    local total=0

    for file in $(find src tests -name "*.lean" -type f); do
        local doc_comments=$(grep -c "/--" "$file" || true)
        local theorems=$(grep -c "^\(theorem\|lemma\|def\) " "$file" || true)

        total=$((total + theorems))
        if [ $theorems -gt 0 ] && [ $doc_comments -gt 0 ]; then
            documented=$((documented + 1))
        fi
    done

    if [ $total -gt 0 ]; then
        local coverage=$((documented * 100 / total))
        print_status "Documentation coverage: $coverage% ($documented/$total items)"
    else
        print_warning "No theorems/definitions found to document"
    fi
}

# Generate recommendations
generate_recommendations() {
    print_header "Recommendations"

    print_status "Generating improvement recommendations..."

    # Check for common issues
    local issues=0

    # Check for files without documentation
    local undocumented_files=$(find src -name "*.lean" -type f ! -exec grep -q "/--" {} \; | wc -l)
    if [ $undocumented_files -gt 0 ]; then
        print_warning "Consider adding documentation to $undocumented_files files"
        issues=$((issues + 1))
    fi

    # Check for very long files
    local long_files=$(find src -name "*.lean" -type f -exec wc -l {} \; | awk '$1 > 300 {print}' | wc -l)
    if [ $long_files -gt 0 ]; then
        print_warning "Consider breaking up $long_files long files (>300 lines)"
        issues=$((issues + 1))
    fi

    # Check for consistent naming
    local inconsistent_names=$(find src -name "*.lean" -type f | grep -v "^src/LeanNiche/" | wc -l)
    if [ $inconsistent_names -gt 0 ]; then
        print_warning "Consider organizing files under consistent module structure"
        issues=$((issues + 1))
    fi

    if [ $issues -eq 0 ]; then
        print_success "No major issues found! Code structure looks good."
    else
        print_status "Found $issues potential areas for improvement"
    fi
}

# Main analysis function
main() {
    print_status "Starting LeanNiche code analysis..."
    echo "=========================================="

    local analysis_type="${1:-all}"

    case "$analysis_type" in
        "structure")
            analyze_structure
            ;;
        "theorems")
            analyze_theorems
            ;;
        "dependencies")
            analyze_dependencies
            ;;
        "complexity")
            analyze_complexity
            ;;
        "documentation")
            analyze_documentation
            ;;
        "recommendations")
            generate_recommendations
            ;;
        "all")
            analyze_structure
            analyze_theorems
            analyze_dependencies
            analyze_complexity
            analyze_documentation
            generate_recommendations
            ;;
        *)
            print_error "Usage: $0 [structure|theorems|dependencies|complexity|documentation|recommendations|all]"
            exit 1
            ;;
    esac

    echo "=========================================="
    print_success "Analysis completed successfully!"
}

# Run main function with all arguments
main "$@"
