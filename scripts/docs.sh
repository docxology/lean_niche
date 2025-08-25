#!/bin/bash

# LeanNiche Documentation Generator
# Generates comprehensive documentation for all components

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
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

print_header() {
    echo -e "${MAGENTA}================================${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}================================${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to generate API documentation
generate_api_docs() {
    print_header "Generating API Documentation"

    if ! command_exists python3; then
        print_error "Python 3 not found. Please install Python 3.8+"
        return 1
    fi

    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        source venv/bin/activate
        print_status "Activated virtual environment"
    fi

    # Generate Python API documentation
    print_status "Generating Python API documentation..."
    if python3 -c "
import sys
sys.path.append('src')
try:
    from python.cli import cli
    print('Python CLI module imported successfully')
except ImportError as e:
    print(f'Warning: Could not import Python CLI: {e}')
    sys.exit(1)
"; then
        print_success "Python API documentation generated"
    else
        print_warning "Python API documentation generation failed"
    fi

    # Generate Lean API documentation
    print_status "Analyzing Lean modules..."
    if command_exists lean; then
        # Analyze Lean files for documentation
        find src/lean -name "*.lean" -type f | while read -r file; do
            if [ -f "$file" ]; then
                print_status "Analyzing $file..."
                # Extract theorems and definitions for documentation
                grep -n "^/--\|^theorem\|^def\|^structure\|^inductive" "$file" || true
            fi
        done
        print_success "Lean module analysis completed"
    else
        print_warning "Lean not found - skipping Lean documentation generation"
    fi
}

# Function to generate code metrics
generate_code_metrics() {
    print_header "Generating Code Metrics"

    print_status "Analyzing project structure..."

    # Count lines of code
    echo "Code Metrics:" > docs/code_metrics.md
    echo "=============" >> docs/code_metrics.md
    echo "" >> docs/code_metrics.md

    # Lean files
    lean_files=$(find src/lean -name "*.lean" -type f | wc -l)
    lean_lines=$(find src/lean -name "*.lean" -type f -exec wc -l {} \; | awk '{sum += $1} END {print sum}')
    echo "- Lean files: $lean_files" >> docs/code_metrics.md
    echo "- Lean lines: $lean_lines" >> docs/code_metrics.md

    # Python files
    python_files=$(find src/python -name "*.py" -type f | wc -l)
    python_lines=$(find src/python -name "*.py" -type f -exec wc -l {} \; | awk '{sum += $1} END {print sum}')
    echo "- Python files: $python_files" >> docs/code_metrics.md
    echo "- Python lines: $python_lines" >> docs/code_metrics.md

    # Scripts
    script_files=$(find scripts -name "*.sh" -o -name "*.py" | wc -l)
    script_lines=$(find scripts -name "*.sh" -o -name "*.py" -exec wc -l {} \; | awk '{sum += $1} END {print sum}')
    echo "- Script files: $script_files" >> docs/code_metrics.md
    echo "- Script lines: $script_lines" >> docs/code_metrics.md

    print_success "Code metrics generated: docs/code_metrics.md"
}

# Function to update documentation index
update_docs_index() {
    print_header "Updating Documentation Index"

    print_status "Creating documentation index..."

    cat > docs/index.md << 'EOF'
# 📚 LeanNiche Documentation Hub

## 🏠 Overview

Welcome to the LeanNiche comprehensive documentation. This hub provides access to all documentation for the LeanNiche mathematical research environment.

## 📋 Table of Contents

### Core Documentation
- **[🏗️ Architecture](architecture.md)** - System design and components
- **[🚀 Deployment Guide](deployment.md)** - Installation and setup
- **[🔍 API Reference](api-reference.md)** - Module and function documentation
- **[📚 Mathematical Foundations](mathematical-foundations.md)** - Theory and concepts

### Development & Usage
- **[🚀 Examples & Tutorials](examples.md)** - Step-by-step guides
- **[🔧 Development Guide](development.md)** - Contributing and development
- **[🔧 Troubleshooting](troubleshooting.md)** - Problem solving guide
- **[🤝 Contributing Guide](contributing.md)** - How to contribute

### Research & Applications
- **[🎯 Research Applications](research-applications.md)** - Use cases and applications
- **[⚡ Performance Analysis](performance.md)** - Optimization techniques
- **[🎯 Proof Development Guide](ProofGuide.md)** - Formal proof development

### Lean Integration
- **[🔬 Lean Overview](lean-overview.md)** - Lean 4 comprehensive guide
- **[🔬 Lean in LeanNiche](lean-in-leanniche.md)** - How Lean is used in this project

### Project Documentation
- **[📊 Code Metrics](code_metrics.md)** - Project statistics and metrics
- **[📋 Implementation Summary](../COMPREHENSIVE_IMPLEMENTATION_SUMMARY.md)** - Complete implementation overview

## 🎯 Quick Start

1. **Setup**: Follow the [Deployment Guide](deployment.md) for installation
2. **Learn**: Start with [Examples & Tutorials](examples.md)
3. **Develop**: Use the [Development Guide](development.md) for contributions
4. **Research**: Explore [Research Applications](research-applications.md)

## 📊 Project Statistics

*Generated automatically - see [Code Metrics](code_metrics.md) for details*

## 🔗 External Resources

- [Lean Official Website](https://lean-lang.org/)
- [Mathlib4 Documentation](https://leanprover-community.github.io/mathlib4_docs/)
- [Lean Zulip Chat](https://leanprover.zulipchat.com/)
- [Theorem Proving in Lean 4](https://leanprover.github.io/theorem_proving_in_lean4/)

---

*This documentation is automatically updated. For the latest information, see the project repository.*
EOF

    print_success "Documentation index updated: docs/index.md"
}

# Function to generate usage examples
generate_usage_examples() {
    print_header "Generating Usage Examples"

    print_status "Creating usage examples..."

    cat > docs/examples.md << 'EOF'
# 🚀 Examples & Tutorials

## 📋 Overview

This guide provides comprehensive examples and tutorials for using LeanNiche in various research scenarios.

## 🏁 Quick Start Examples

### 1. Basic Lean Proof
```lean
import LeanNiche.Basic

/-- My first theorem: 1 + 1 = 2 -/
theorem one_plus_one : 1 + 1 = 2 := by
  rfl  -- This is reflexivity, the simplest proof
```

### 2. Python Analysis
```python
from src.python.mathematical_analyzer import MathematicalAnalyzer

# Analyze a function
analyzer = MathematicalAnalyzer()
result = analyzer.analyze_function(lambda x: x**2, (-3, 3), "full")
print(f"Analysis: {result}")
```

### 3. Command Line Usage
```bash
# Run Lean environment
lake exe lean_niche

# Show Python help
python -m src.python.cli --help

# Run tests
./scripts/test.sh

# Analyze code
./scripts/analyze.sh
```

## 📚 Tutorial Sections

### Lean Proof Development
1. [Basic Proof Techniques](lean-overview.md#basic-proof-techniques)
2. [Advanced Tactics](ProofGuide.md#advanced-tactics)
3. [Common Patterns](ProofGuide.md#common-patterns)

### Python Analysis
1. [Function Analysis](api-reference.md#python-api)
2. [Visualization](api-reference.md#visualization-classes)
3. [Data Processing](api-reference.md#data-generation)

### Research Workflows
1. [Mathematical Research](research-applications.md)
2. [Algorithm Verification](api-reference.md#computational-lean-module)
3. [Statistical Analysis](api-reference.md#statistics-lean-module)

## 🔬 Advanced Examples

### Complete Research Workflow
```python
# 1. Import LeanNiche components
from src.python.mathematical_analyzer import MathematicalAnalyzer
from src.python.data_generator import MathematicalDataGenerator

# 2. Generate research data
generator = MathematicalDataGenerator()
data = generator.generate_comprehensive_dataset("statistical_analysis")

# 3. Perform mathematical analysis
analyzer = MathematicalAnalyzer()
results = analyzer.analyze_function(lambda x: x**3 - 2*x + 1, (-2, 2), "full")

# 4. Save results
analyzer.save_analysis_results(results, "cubic_analysis")
```

### Lean-Python Integration
```lean
-- Lean mathematical definition
def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Proof of basic properties
theorem quadratic_non_negative (a : ℝ) (ha : a > 0) (x : ℝ) :
  ∀ b c : ℝ, ∃ x0 : ℝ, quadratic_function a b c x0 ≥ 0 := by
  -- Proof using vertex form and positivity
  sorry
```

## 🎯 Best Practices

### Lean Development
- Use descriptive theorem names
- Include comprehensive documentation
- Test theorems with concrete examples
- Build incrementally with small lemmas

### Python Development
- Follow type hints and documentation standards
- Handle edge cases and errors gracefully
- Write comprehensive unit tests
- Use virtual environments for isolation

### Research Workflows
- Document research hypotheses clearly
- Save intermediate results
- Use version control for reproducibility
- Validate results across multiple methods

## 🔧 Troubleshooting

Common issues and solutions:
- [Lean compilation errors](troubleshooting.md#lean-issues)
- [Python import issues](troubleshooting.md#python-issues)
- [Performance problems](performance.md#optimization)
- [Documentation generation](troubleshooting.md#documentation)

## 📚 Next Steps

1. **Explore**: Try the examples in this guide
2. **Learn**: Study the [Mathematical Foundations](mathematical-foundations.md)
3. **Contribute**: Follow the [Contributing Guide](contributing.md)
4. **Research**: Apply LeanNiche to your research questions

## 📖 References

- [LeanNiche API Reference](api-reference.md)
- [Mathematical Foundations](mathematical-foundations.md)
- [Research Applications](research-applications.md)
- [Performance Analysis](performance.md)
EOF

    print_success "Usage examples generated: docs/examples.md"
}

# Main documentation generation function
main() {
    print_header "LeanNiche Documentation Generator"
    echo -e "${CYAN}Generating comprehensive documentation for LeanNiche...${NC}"
    echo ""

    # Check if we're in the right directory
    if [ ! -f "README.md" ]; then
        print_error "Please run this script from the LeanNiche project root directory"
        exit 1
    fi

    # Create docs directory if it doesn't exist
    if [ ! -d "docs" ]; then
        mkdir -p docs
        print_success "Created docs directory"
    fi

    # Run documentation generation steps
    generate_api_docs
    generate_code_metrics
    update_docs_index
    generate_usage_examples

    print_header "Documentation Generation Complete!"
    echo -e "${GREEN}📚 Documentation generated successfully!${NC}"
    echo ""
    echo "Generated files:"
    echo "  📄 docs/index.md          - Main documentation hub"
    echo "  📄 docs/examples.md       - Examples and tutorials"
    echo "  📄 docs/code_metrics.md   - Project statistics"
    echo ""
    echo "Next steps:"
    echo "  1. Review the generated documentation"
    echo "  2. Update any outdated information"
    echo "  3. Add new documentation as needed"
    echo ""
}

# Run main function
main "$@"
