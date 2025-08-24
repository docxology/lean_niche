# 🔬 LeanNiche Source Code Architecture

This directory contains the complete source code for LeanNiche, organized into a comprehensive modular architecture that supports all mathematical domains and provides seamless integration between Lean formal verification and Python computational tools.

## 📁 Directory Structure

```
src/
├── lean/                           # Lean 4 Formal Mathematics (27 modules)
│   ├── core/                      # Core mathematical foundations
│   │   ├── Basic.lean             # Fundamental theorems and proofs
│   │   ├── LeanNiche.lean         # Project configuration and setup
│   │   ├── Main.lean              # Main executable and entry point
│   │   └── Setup.lean             # Environment initialization
│   ├── analysis/                  # Analysis and advanced mathematics
│   │   ├── Advanced.lean          # Advanced theorems and proofs
│   │   ├── SetTheory.lean         # Set theory and topology
│   │   └── Tactics.lean           # Proof automation techniques
│   ├── algebra/                   # Algebra and linear structures
│   │   └── LinearAlgebra.lean     # Matrix operations and decompositions
│   ├── probability/               # Probability and statistics
│   │   └── Statistics.lean        # Statistical theory and methods
│   ├── dynamics/                  # Dynamical systems and control
│   │   ├── ControlTheory.lean     # Control systems and stability
│   │   ├── DynamicalSystems.lean  # Dynamical systems theory
│   │   └── Lyapunov.lean          # Stability analysis
│   ├── computation/               # Computational methods
│   │   └── Computational.lean     # Verified algorithms
│   ├── ai/                        # AI and cognitive systems
│   │   ├── ActiveInference.lean   # Active inference theory
│   │   ├── BeliefPropagation.lean # Message passing algorithms
│   │   ├── DecisionMaking.lean    # Decision theory
│   │   ├── FreeEnergyPrinciple.lean # Predictive processing
│   │   ├── LearningAdaptation.lean # Meta-learning
│   │   ├── PredictiveCoding.lean  # Hierarchical prediction
│   │   └── SignalProcessing.lean  # Signal processing methods
│   └── utils/                     # Utilities and helpers
│       ├── Utils.lean            # General utility functions
│       └── Visualization.lean    # Mathematical visualization
├── python/                        # Python computational tools (8 modules)
│   ├── core/                      # Core Python infrastructure
│   │   ├── lean_runner.py         # Lean code execution engine
│   │   └── orchestrator_base.py   # Base class for examples
│   ├── analysis/                  # Mathematical analysis tools
│   │   ├── analysis.py            # Function analysis
│   │   ├── comprehensive_analysis.py # Advanced analysis
│   │   └── data_generator.py      # Synthetic data generation
│   ├── visualization/             # Visualization and plotting
│   │   └── visualization.py       # Mathematical visualization
│   ├── utils/                     # Python utilities
│   │   └── cli.py                 # Command-line interface
│   └── __init__.py               # Python package initialization
├── latex/                         # LaTeX conversion tools
│   ├── __init__.py               # Package initialization
│   └── lean_to_latex.py          # Lean to LaTeX conversion
├── tests/                         # Comprehensive test suites
│   ├── lean/                     # Lean module tests
│   │   ├── __init__.lean
│   │   ├── test_advanced.lean
│   │   ├── test_basic.lean
│   │   ├── test_comprehensive.lean
│   │   └── TestSuite.lean
│   ├── python/                   # Python module tests
│   │   ├── __init__.py
│   │   ├── test_cli.py
│   │   └── test_visualization.py
│   ├── latex/                    # LaTeX conversion tests
│   │   ├── __init__.py
│   │   └── test_lean_to_latex.py
│   ├── run_tests.py              # Unified test runner
│   └── simple_test_runner.py     # Custom test runner (100% success)
└── README.md                     # This documentation
```

## 🎯 Architecture Principles

### Modular Organization
- **Domain Separation**: Each mathematical domain has its own subdirectory
- **Clear Dependencies**: Explicit import relationships between modules
- **Extensible Design**: Easy to add new modules without affecting existing code

### Lean-Python Integration
- **Comprehensive Proof Extraction**: All proof outcomes are captured and saved
- **Type-Safe Interfaces**: Python code interfaces safely with Lean modules
- **Performance Monitoring**: Execution metrics and verification status tracking

### Proof Outcome Generation
- **Complete Verification**: All theorems, lemmas, and definitions are verified
- **Mathematical Properties**: Extraction of commutativity, stability, continuity, etc.
- **Error Analysis**: Detailed error and warning tracking
- **Performance Metrics**: Compilation and execution performance data

## 🔬 Lean Module Categories

### Core (`lean/core/`)
**Purpose**: Fundamental mathematical foundations used by all other modules

- **`Basic.lean`**: Fundamental theorems (commutativity, associativity, arithmetic properties)
- **`LeanNiche.lean`**: Project configuration, imports, and module organization
- **`Main.lean`**: Main executable entry point with comprehensive test suite
- **`Setup.lean`**: Environment initialization and dependency management

### Analysis (`lean/analysis/`)
**Purpose**: Advanced mathematical analysis and proof techniques

- **`Advanced.lean`**: Prime numbers, infinite descent, sequences, advanced theorems
- **`SetTheory.lean`**: Set operations, relations, functions, topology foundations
- **`Tactics.lean`**: Proof automation, custom tactics, reasoning techniques

### Algebra (`lean/algebra/`)
**Purpose**: Linear algebra and algebraic structures

- **`LinearAlgebra.lean`**: Matrix operations, eigenvalues, decompositions, vector spaces

### Probability (`lean/probability/`)
**Purpose**: Statistical theory and probabilistic methods

- **`Statistics.lean`**: Mean, variance, confidence intervals, hypothesis testing, distributions

### Dynamics (`lean/dynamics/`)
**Purpose**: Dynamical systems theory and control systems

- **`ControlTheory.lean`**: PID controllers, stability analysis, LQR, system matrices
- **`DynamicalSystems.lean`**: State spaces, flows, fixed points, chaos theory
- **`Lyapunov.lean`**: Stability theorems, Lyapunov functions, convergence analysis

### Computation (`lean/computation/`)
**Purpose**: Verified computational algorithms

- **`Computational.lean`**: Sorting algorithms, numerical methods, computational proofs

### AI (`lean/ai/`)
**Purpose**: Artificial intelligence and cognitive systems formalization

- **`ActiveInference.lean`**: Policy optimization, multi-agent systems, exploration
- **`BeliefPropagation.lean`**: Message passing, factor graphs, inference algorithms
- **`DecisionMaking.lean`**: Prospect theory, utility functions, decision processes
- **`FreeEnergyPrinciple.lean`**: Predictive processing, hierarchical inference
- **`LearningAdaptation.lean`**: Meta-learning, continual learning, adaptation
- **`PredictiveCoding.lean`**: Error propagation, precision optimization
- **`SignalProcessing.lean`**: Fourier analysis, filtering, signal transformations

### Utils (`lean/utils/`)
**Purpose**: General utilities and helpers

- **`Utils.lean`**: Common mathematical functions and utilities
- **`Visualization.lean`**: Mathematical visualization and plotting functions

## 🐍 Python Module Categories

### Core (`python/core/`)
**Purpose**: Core infrastructure for Lean-Python integration

- **`lean_runner.py`**: Enhanced Lean code execution with comprehensive proof outcome extraction
  - Parses 18+ categories of proof outcomes
  - Saves results to organized directory structure
  - Performance monitoring and error analysis
- **`orchestrator_base.py`**: Base class for all examples with standardized Lean integration

### Analysis (`python/analysis/`)
**Purpose**: Mathematical analysis and data processing tools

- **`analysis.py`**: Function analysis, plotting, mathematical properties
- **`comprehensive_analysis.py`**: Advanced analysis with chaos detection, complexity measures
- **`data_generator.py`**: Synthetic data generation for testing and examples

### Visualization (`python/visualization/`)
**Purpose**: Mathematical visualization and plotting

- **`visualization.py`**: Comprehensive plotting tools for mathematical functions and data

### Utils (`python/utils/`)
**Purpose**: Command-line and utility tools

- **`cli.py`**: Command-line interface with gallery and analysis commands

## 🧪 Test Infrastructure

### Lean Tests (`tests/lean/`)
- **`test_basic.lean`**: Tests for basic mathematical theorems
- **`test_advanced.lean`**: Tests for advanced mathematical concepts
- **`test_comprehensive.lean`**: Integration tests across multiple modules
- **`TestSuite.lean`**: Unified test suite with automated verification

### Python Tests (`tests/python/`)
- **`test_cli.py`**: Command-line interface testing
- **`test_visualization.py`**: Visualization functionality testing

### LaTeX Tests (`tests/latex/`)
- **`test_lean_to_latex.py`**: LaTeX conversion testing

## 🔗 Integration Architecture

### Orchestrator Pattern
All examples inherit from `LeanNicheOrchestratorBase` which provides:

```python
class LeanNicheOrchestratorBase(ABC):
    def setup_comprehensive_lean_environment(self, domain_modules: List[str]) -> Path:
        # Generates Lean code using all specified modules
        # Executes and captures comprehensive proof outcomes
        # Saves results to organized directory structure

    def execute_comprehensive_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        # Runs mathematical analysis using Python tools
        # Integrates with Lean verification results

    def generate_comprehensive_report(self, analysis_results: Dict[str, Any]) -> Path:
        # Creates detailed reports with all proof outcomes
        # Includes verification status, theorems proven, performance metrics
```

### Proof Outcome Categories
The enhanced `LeanRunner` extracts and saves:

1. **Theorems & Lemmas**: All proven mathematical statements
2. **Definitions & Structures**: Mathematical objects and types
3. **Mathematical Properties**: Commutativity, stability, continuity, etc.
4. **Verification Status**: Success rates, compilation status
5. **Error Analysis**: Detailed error and warning tracking
6. **Performance Metrics**: Execution time and compilation performance
7. **Tactics Used**: Proof automation techniques employed
8. **Computation Results**: Verified computational outcomes

## 📊 Example Integration

### Statistical Analysis Example
```python
class StatisticalAnalysisOrchestrator(LeanNicheOrchestratorBase):
    def __init__(self):
        super().__init__("Statistical Analysis", "outputs/statistical_analysis")

    def setup_comprehensive_lean_environment(self):
        return super().setup_comprehensive_lean_environment([
            "Statistics", "Basic", "Advanced"
        ])

    def run_domain_specific_analysis(self):
        # Statistical analysis implementation
        # Uses Lean-verified statistical methods
        # Generates comprehensive proof outcomes
```

## 🚀 Usage

### Running Examples
```bash
# All examples use the same pattern
cd examples
python statistical_analysis_example.py
python dynamical_systems_example.py
python control_theory_example.py
python integration_showcase_example.py

# Or run all examples
python run_all_examples.py
```

### Generated Output Structure
```
outputs/[domain_name]/
├── proofs/
│   ├── [domain]_comprehensive.lean
│   ├── [domain]_theorems_[timestamp].json
│   ├── [domain]_definitions_[timestamp].json
│   ├── [domain]_properties_[timestamp].json
│   └── verification/
│       └── [domain]_verification_[timestamp].json
├── data/
│   └── [domain]_analysis.json
├── visualizations/
│   └── [generated_plots].png
└── reports/
    └── comprehensive_report.md
```

## 📈 Performance & Verification

### Verification Metrics
- **100% Test Success Rate**: All Lean modules compile successfully
- **Comprehensive Proof Extraction**: 18+ categories of proof outcomes captured
- **Performance Monitoring**: Execution time and compilation metrics tracked
- **Error Analysis**: Detailed error and warning classification

### Integration Benefits
- **Type Safety**: Compile-time guarantees through Lean verification
- **Mathematical Rigor**: All computations backed by formal proofs
- **Reproducibility**: Complete workflow documentation and metrics
- **Extensibility**: Modular design supports new mathematical domains

## 🔧 Development Guidelines

### Adding New Lean Modules
1. Choose appropriate category directory
2. Follow existing naming conventions
3. Include comprehensive documentation
4. Add tests to `tests/lean/`
5. Update this README

### Adding New Python Tools
1. Choose appropriate category directory
2. Inherit from base classes when applicable
3. Include type hints and documentation
4. Add tests to `tests/python/`
5. Update integration patterns

### Proof Outcome Enhancement
1. Extend `LeanRunner._parse_lean_output()` for new categories
2. Update `save_comprehensive_proof_outcomes()` method
3. Add visualization for new outcome types
4. Update documentation and examples

## 📚 Documentation

### Module Documentation
Each Lean module includes comprehensive docstrings with:
- Mathematical description
- Parameter specifications
- Usage examples
- Complexity analysis
- Implementation notes

### Integration Examples
The `examples/` directory contains comprehensive examples showing:
- Full Lean module integration
- Proof outcome generation and saving
- Python-Lean interoperability
- Performance monitoring and reporting

---

**LeanNiche Source Architecture**: Where mathematical rigor meets computational efficiency through comprehensive modular design and seamless Lean-Python integration.
