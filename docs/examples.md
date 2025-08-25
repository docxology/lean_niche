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
