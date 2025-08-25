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

# Run comprehensive analysis
python src/python/analysis/comprehensive_analysis.py --gallery

# Run tests
python src/tests/simple_test_runner.py

# Analyze code and generate reports
./scripts/analyze.sh

# Generate visualization gallery
python -c "
from src.python.analysis.comprehensive_analysis import ComprehensiveMathematicalAnalyzer
analyzer = ComprehensiveMathematicalAnalyzer()
results = analyzer.create_analysis_gallery()
print('Gallery created successfully!')
"
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
from src.python.analysis.comprehensive_analysis import ComprehensiveMathematicalAnalyzer
from src.python.data_generation.mathematical_data_generator import MathematicalDataGenerator
from src.python.visualization.mathematical_visualizer import MathematicalVisualizer

# 2. Initialize components
analyzer = ComprehensiveMathematicalAnalyzer(
    output_dir="research_output",
    log_level="INFO",
    enable_logging=True
)
generator = MathematicalDataGenerator()
visualizer = MathematicalVisualizer()

# 3. Generate research data
data = generator.generate_comprehensive_dataset("statistical_analysis")
print(f"Generated {len(data)} data points")

# 4. Perform comprehensive mathematical analysis
def research_function(x):
    """Research function: damped oscillator"""
    import numpy as np
    return np.exp(-0.1 * x) * np.sin(2 * np.pi * 0.5 * x)

results = analyzer.comprehensive_function_analysis(
    research_function, 
    (0, 20), 
    "full"
)

# 5. Create visualizations
fig = visualizer.plot_function(
    research_function,
    (0, 20),
    "Damped Oscillator Analysis",
    save_path="research_output/damped_oscillator.png"
)

# 6. Generate comprehensive report
report_path = analyzer.save_analysis_results(results, "damped_oscillator_analysis")
print(f"Research results saved to: {report_path}")

# 7. Create analysis gallery
gallery_results = analyzer.create_analysis_gallery()
print(f"Analysis gallery created with {len(gallery_results)} examples")
```

### Advanced Statistical Analysis
```python
# Statistical research example
from src.python.analysis.comprehensive_analysis import ComprehensiveMathematicalAnalyzer
import numpy as np

# Initialize analyzer
analyzer = ComprehensiveMathematicalAnalyzer()

# Generate sample data for statistical analysis
np.random.seed(42)
sample_data = {
    'normal_data': np.random.normal(0, 1, 1000),
    'exponential_data': np.random.exponential(2, 1000),
    'uniform_data': np.random.uniform(-3, 3, 1000)
}

# Perform statistical analysis on each dataset
for name, data in sample_data.items():
    print(f"\n=== Analyzing {name} ===")
    
    # Convert to function for analysis
    def data_function(x):
        # Create empirical CDF
        sorted_data = np.sort(data)
        return np.searchsorted(sorted_data, x) / len(sorted_data)
    
    # Analyze the empirical distribution
    results = analyzer.comprehensive_function_analysis(
        data_function,
        (np.min(data), np.max(data)),
        "statistical"
    )
    
    # Print key statistics
    if 'statistical' in results:
        stats = results['statistical']
        print(f"Mean: {stats.get('mean', 'N/A'):.4f}")
        print(f"Std: {stats.get('std', 'N/A'):.4f}")
        print(f"Skewness: {stats.get('skewness', 'N/A'):.4f}")
        print(f"Kurtosis: {stats.get('kurtosis', 'N/A'):.4f}")
        
        if 'distribution_fit' in stats and 'best_fit' in stats['distribution_fit']:
            print(f"Best distribution fit: {stats['distribution_fit']['best_fit']}")

print("\n✅ Statistical analysis complete!")
```

### Dynamical Systems Research
```python
# Dynamical systems analysis example
import numpy as np
from src.python.analysis.comprehensive_analysis import ComprehensiveMathematicalAnalyzer
from src.python.visualization.dynamical_systems_visualizer import DynamicalSystemsVisualizer

# Initialize components
analyzer = ComprehensiveMathematicalAnalyzer()
dyn_viz = DynamicalSystemsVisualizer()

# Define a dynamical system: Lorenz attractor projection
def lorenz_x_component(t):
    """X-component of Lorenz system solution"""
    # Simplified approximation for demonstration
    return 10 * np.sin(0.5 * t) * np.exp(-0.01 * t)

# Analyze the dynamical behavior
print("🔄 Analyzing Lorenz system X-component...")
results = analyzer.comprehensive_function_analysis(
    lorenz_x_component,
    (0, 100),
    "full"
)

# Check for chaotic behavior indicators
if 'advanced' in results:
    advanced = results['advanced']
    lyapunov = advanced.get('lyapunov_exponent', 0)
    fractal_dim = advanced.get('fractal_dimension', 1)
    
    print(f"Lyapunov exponent: {lyapunov:.6f}")
    print(f"Fractal dimension: {fractal_dim:.4f}")
    
    if lyapunov > 0:
        print("🌀 Chaotic behavior detected!")
    else:
        print("📊 Regular behavior detected")

# Create phase space visualization
trajectory_data = [(lorenz_x_component(t), lorenz_x_component(t + 0.1)) 
                   for t in np.linspace(0, 50, 1000)]

phase_fig = dyn_viz.plot_trajectory(
    trajectory_data,
    "Lorenz System Phase Space",
    save_path="research_output/lorenz_phase_space.png"
)

print("✅ Dynamical systems analysis complete!")
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
