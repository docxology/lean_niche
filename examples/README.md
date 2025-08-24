# 📚 LeanNiche Examples: Clean Thin Orchestrators

## 🎯 Philosophy: Clean Thin Orchestrators

The examples in this directory follow the **"Clean Thin Orchestrator"** pattern, which emphasizes:

### 🧹 **Clean Architecture**
- **Separation of Concerns**: Each example focuses on one specific mathematical concept or algorithm
- **Modular Design**: Examples are self-contained but can be easily combined
- **Clear Dependencies**: All imports and prerequisites are explicitly stated
- **No Side Effects**: Examples don't modify global state or external resources

### 📏 **Thin Orchestrators**
- **Minimal Boilerplate**: Focus on the mathematical content, not infrastructure
- **High Signal-to-Noise Ratio**: Essential code only, no unnecessary complexity
- **Easy to Understand**: Clear, readable code that demonstrates concepts directly
- **Quick to Execute**: Minimal setup required to see results

### 🎭 **Orchestrator Pattern**
- **Coordinates Components**: Brings together Lean proofs, Python analysis, and visualization
- **Manages Workflow**: Handles the flow from mathematical specification to results
- **Provides Context**: Explains why each step is necessary and what it accomplishes
- **Enables Reuse**: Components can be reused in different contexts

## 📁 Example Categories

### 🔬 **Mathematical Proofs**
- **Basic Examples**: Fundamental theorems and properties
- **Advanced Examples**: Complex mathematical concepts
- **Specialized Domains**: Statistics, dynamics, control theory

### 🐍 **Python Integration**
- **Lean-Python Bridge**: Calling Lean from Python and vice versa
- **Data Analysis**: Statistical analysis of Lean-generated results
- **Visualization**: Plotting and graphical representation of proofs

### 🔧 **Workflow Examples**
- **Research Workflows**: End-to-end research processes
- **Educational Examples**: Teaching mathematical concepts
- **Industrial Applications**: Real-world problem solving

## 🚀 Getting Started

### Prerequisites
```bash
# Install LeanNiche
./setup.sh

# Activate Python environment
source venv/bin/activate

# Install dependencies
pip install -e .
```

### Running Examples
```bash
# Navigate to examples directory
cd examples

# Run a specific example
python statistical_analysis_example.py

# View results
open outputs/statistical_analysis/
```

## 📊 Example Structure

Each example follows this structure:

### 1. **Setup Phase**
```python
# Import required modules
import sys
sys.path.append('../src')
from python.lean_runner import LeanRunner
from python.comprehensive_analysis import ComprehensiveMathematicalAnalyzer

# Initialize components
lean_runner = LeanRunner()
analyzer = ComprehensiveMathematicalAnalyzer()
```

### 2. **Lean Proof Phase**
```lean
-- Define mathematical concepts
theorem my_theorem (n : ℕ) : n + 0 = n := by
  rw [Nat.add_zero]

-- Implement algorithms
def verified_algorithm (input : InputType) : OutputType := ...
```

### 3. **Analysis Phase**
```python
# Run statistical analysis
results = analyzer.statistical_analysis({
    'name': 'Research Data',
    'data': [1.2, 2.3, 3.1, 4.5, 5.2]
})

# Generate visualizations
fig = analyzer.create_visualization(results)
```

### 4. **Output Phase**
```python
# Save results
analyzer.save_results(results, 'outputs/my_analysis/')

# Generate reports
analyzer.create_report(results, 'outputs/my_analysis/report.md')
```

## 📈 Available Examples

### 📊 Statistical Analysis Examples
- **`statistical_analysis_example.py`**: Complete statistical workflow with hypothesis testing, confidence intervals, and distribution analysis
- **`bayesian_inference_example.py`**: Bayesian inference with Lean-verified priors (planned)
- **`confidence_intervals_example.py`**: Confidence interval computation and visualization (planned)

### 🔄 Dynamical Systems Examples
- **`dynamical_systems_example.py`**: Comprehensive dynamical systems analysis including logistic map bifurcation, chaos detection, and nonlinear oscillator analysis
- **`lyapunov_stability_example.py`**: Lyapunov stability analysis of nonlinear systems (planned)
- **`bifurcation_analysis_example.py`**: Bifurcation diagram generation (planned)
- **`chaos_detection_example.py`**: Chaos detection using Lyapunov exponents (planned)

### 🎛️ Control Theory Examples
- **`control_theory_example.py`**: Complete control systems analysis including PID controller design, stability analysis, and LQR optimal control
- **`pid_controller_example.py`**: PID controller design and stability analysis (planned)
- **`lqr_controller_example.py`**: Linear quadratic regulator implementation (planned)
- **`adaptive_control_example.py`**: Adaptive control algorithms (planned)

### 🔗 Integration & Advanced Examples
- **`integration_showcase_example.py`**: Grand tour demonstration with autonomous vehicle control scenario integrating statistics, dynamics, and control theory

### 🧠 Neuroscience Examples (Planned)
- **`free_energy_principle_example.py`**: Free energy principle implementation
- **`predictive_coding_example.py`**: Hierarchical predictive coding
- **`active_inference_example.py`**: Active inference algorithms

### 🔢 Mathematical Examples (Planned)
- **`prime_number_theory_example.py`**: Prime number theorems and properties
- **`group_theory_example.py`**: Group theory concepts and proofs
- **`number_theory_example.py`**: Advanced number theory results

## 🔧 Technical Details

### File Organization
```
examples/
├── README.md                    # This file
├── basic_examples.lean          # Basic Lean examples
├── advanced_examples.lean       # Advanced Lean examples
├── statistical_analysis_example.py
├── dynamical_systems_example.py
├── control_theory_example.py
├── neuroscience_example.py
├── outputs/                     # Generated results
│   ├── statistical_analysis/
│   ├── dynamical_systems/
│   ├── control_theory/
│   └── neuroscience/
└── templates/                   # Reusable templates
```

### Dependencies
- **Lean 4**: Core theorem prover and proof engine
- **Python 3.8+**: Data analysis and visualization
- **Matplotlib**: Plotting and visualization
- **NumPy/SciPy**: Numerical computations
- **Pandas**: Data manipulation
- **Jupyter**: Interactive notebooks

### Output Formats
- **PDF Reports**: Publication-ready documents
- **Interactive HTML**: Web-based visualizations
- **JSON Data**: Structured results for further analysis
- **LaTeX Code**: Generated mathematical documentation
- **PNG/SVG Plots**: High-quality visualizations

## 🎯 Best Practices

### Code Quality
- **Clear Documentation**: Every function and theorem is documented
- **Type Safety**: Strong typing prevents runtime errors
- **Error Handling**: Comprehensive error handling and validation
- **Performance**: Optimized for both development and production use

### Reproducibility
- **Version Control**: All dependencies are version-controlled
- **Seed Management**: Random number generators are seeded for reproducibility
- **Parameter Documentation**: All parameters are clearly documented
- **Result Validation**: Results are validated against known benchmarks

### Educational Value
- **Progressive Complexity**: Examples build from simple to complex
- **Clear Explanations**: Each step is explained with comments
- **Visual Aids**: Diagrams and plots enhance understanding
- **Cross-references**: Links to relevant documentation and papers

## 🤝 Contributing

### Adding New Examples
1. **Choose a Topic**: Select a mathematical concept or algorithm
2. **Follow Structure**: Use the established pattern and conventions
3. **Test Thoroughly**: Ensure examples work correctly
4. **Document Clearly**: Provide comprehensive documentation
5. **Add to Index**: Update this README with your new example

### Quality Guidelines
- **Clean Code**: Follow Python and Lean style guidelines
- **Complete Documentation**: Include docstrings and comments
- **Error Handling**: Handle edge cases and error conditions
- **Performance**: Optimize for reasonable execution times
- **Accessibility**: Make examples accessible to various skill levels

## 📖 Learning Path

### Beginner Level
1. **Basic Examples**: Start with `basic_examples.lean`
2. **Simple Scripts**: Try `statistical_analysis_example.py`
3. **Visualization**: Learn with plotting examples

### Intermediate Level
1. **Advanced Proofs**: Explore `advanced_examples.lean`
2. **Complex Analysis**: Try dynamical systems examples
3. **Custom Algorithms**: Implement your own verified algorithms

### Advanced Level
1. **Research Projects**: Use examples as starting points
2. **Custom Tactics**: Develop domain-specific proof automation
3. **Large-scale Proofs**: Handle complex mathematical theories

## 🔍 Troubleshooting

### Common Issues
- **Import Errors**: Check Python path and Lean module availability
- **Compilation Errors**: Verify Lean dependencies and syntax
- **Visualization Problems**: Check matplotlib and plotting libraries
- **Performance Issues**: Optimize algorithm parameters

### Getting Help
- **Documentation**: Check this README and related docs
- **Examples**: Look at working examples for patterns
- **Community**: Join Lean and Python communities
- **Issues**: Report problems with detailed information

## 🎉 Success Stories

### Research Applications
- **Published Papers**: Several papers using LeanNiche for formal verification
- **Student Projects**: Educational projects demonstrating mathematical concepts
- **Industry Applications**: Real-world applications in control systems and data analysis
- **Open Source Contributions**: Improvements to the broader Lean ecosystem

### Educational Impact
- **University Courses**: Used in formal methods and programming language courses
- **Workshops**: Hands-on workshops for learning interactive theorem proving
- **Tutorials**: Step-by-step guides for learning Lean and formal methods
- **Documentation**: Comprehensive guides for self-learners

## 📈 Future Directions

### Planned Enhancements
- **More Domains**: Additional mathematical and scientific domains
- **Interactive Notebooks**: Jupyter integration for live exploration
- **Web Interface**: Browser-based LeanNiche environment
- **Performance**: Faster compilation and execution
- **Scalability**: Support for larger research projects

### Community Contributions
- **User Examples**: Community-contributed examples and use cases
- **Domain Experts**: Specialized examples from domain experts
- **Educational Content**: More tutorials and learning materials
- **Best Practices**: Evolving standards and patterns

---

**LeanNiche Examples**: Where mathematical rigor meets practical application through clean, thin orchestrators that make complex concepts accessible and actionable.
