# LeanNiche Test Suite

This directory contains the comprehensive test suite for the LeanNiche mathematical research environment. The test suite is organized by module type and provides thorough testing for all components.

## 🏗️ Test Structure

```
src/tests/
├── run_tests.py              # Unified test runner (main entry point)
├── README.md                 # This file
├── TestSuite.lean           # Legacy Lean test suite
├── python/                  # Python module tests
│   ├── __init__.py
│   ├── test_cli.py         # CLI command tests
│   └── test_visualization.py # Visualization module tests
├── latex/                   # LaTeX conversion tests
│   ├── __init__.py
│   └── test_lean_to_latex.py # LaTeX conversion tests
└── lean/                    # Lean module tests
    ├── __init__.lean
    ├── test_basic.lean      # Basic.lean tests
    ├── test_advanced.lean   # Advanced.lean tests
    └── test_comprehensive.lean # Multi-module tests
```

## 📋 Test Coverage

### Python Tests (`python/`)
- **CLI Module**: Tests for command-line interface commands
  - Help command functionality
  - Plot function command
  - Analyze data command
  - Gallery command
  - Error handling and edge cases

- **Visualization Module**: Tests for mathematical visualization
  - Function plotting
  - Statistical data visualization
  - Interactive plots
  - Network visualization
  - Trajectory plotting
  - Phase portrait generation
  - Bifurcation diagrams
  - Gallery creation

### LaTeX Tests (`latex/`)
- **Lean to LaTeX Converter**: Tests for mathematical notation conversion
  - Symbol mapping (∧, →, ∀, ∈, etc.)
  - Expression conversion
  - Theorem conversion
  - Definition conversion
  - Namespace handling
  - File conversion (complete .lean to .tex)
  - Error handling and edge cases

### Lean Tests (`lean/`)
- **Basic Module**: Tests for fundamental mathematical concepts
  - Identity function
  - Addition commutativity/associativity
  - Multiplication properties
  - Function composition
  - Injective/surjective functions

- **Advanced Module**: Tests for advanced mathematical concepts
  - Factorial function and theorems
  - Natural number theorems
  - List reversal operations
  - Sum functions
  - Prime number properties
  - Number theory theorems

- **Comprehensive Tests**: Multi-module integration tests
  - Computational module (Fibonacci, list operations)
  - Tactics module (proof automation)
  - Utils module (mathematical utilities)
  - Setup module (configuration)
  - Module import verification
  - Integration testing

## 🚀 Running Tests

### Quick Start
```bash
# Run all tests
python src/tests/run_tests.py

# Run only Python tests
python src/tests/run_tests.py --python-only

# Run only Lean tests
python src/tests/run_tests.py --lean-only

# Run only LaTeX tests
python src/tests/run_tests.py --latex-only

# Run with coverage report
python src/tests/run_tests.py --coverage

# Run in parallel (Python tests only)
python src/tests/run_tests.py --parallel

# Verbose output
python src/tests/run_tests.py --verbose

# Stop on first failure
python src/tests/run_tests.py --fail-fast

# Generate JUnit XML report
python src/tests/run_tests.py --junit-xml test_results.xml
```

### Using Make Commands
```bash
# Run all tests
make test

# Run only Python tests
python src/tests/run_tests.py --python-only

# Run with coverage
make coverage

# Performance testing
python src/tests/run_tests.py --performance-only
```

### Using pytest Directly
```bash
# Python tests only
pytest src/tests/python/ -v

# LaTeX tests only
pytest src/tests/latex/ -v

# With coverage
pytest src/tests/python/ src/tests/latex/ --cov=src --cov-report=html
```

### Using Lake for Lean Tests
```bash
# Run Lean tests through Lake
lake exe lean_niche

# Build and test
lake build && lake exe lean_niche
```

## 🧪 Test Types

### Unit Tests
- Test individual functions and methods
- Verify correct behavior with various inputs
- Check error handling and edge cases
- Mock external dependencies

### Integration Tests
- Test module interactions
- Verify import dependencies
- Check configuration loading
- Test end-to-end workflows

### Performance Tests
- Measure execution time
- Monitor memory usage
- Test scalability
- Benchmark critical functions

### Coverage Tests
- Track code coverage
- Identify untested code paths
- Generate coverage reports
- Ensure comprehensive testing

## 📊 Test Results

### Expected Output
```
🔍 LeanNiche Comprehensive Test Suite
============================================================
Testing modules: python, lean, latex, integration, performance

📋 Python Module Tests
✅ Python tests passed

📋 Lean Module Tests
✅ Lean tests passed

📋 LaTeX Conversion Tests
✅ LaTeX tests passed

📋 Integration Tests
✅ Integration tests passed

📋 Performance Tests
✅ Performance tests passed

📋 Coverage Analysis
✅ Coverage report generated

📊 Test Summary
Total test time: 45.23 seconds
Tests completed: 5
Tests passed: 5
Tests failed: 0
Success rate: 100.0%

  python        ✅ PASSED
  lean          ✅ PASSED
  latex         ✅ PASSED
  integration   ✅ PASSED
  performance   ✅ PASSED

🎉 All tests completed successfully!
```

### Coverage Report
When running with `--coverage`, coverage reports are generated in:
- `htmlcov/` - HTML coverage report (open `htmlcov/index.html`)
- `coverage.xml` - XML coverage report for CI/CD integration
- Terminal output shows missing lines

## 🔧 Test Configuration

### pytest Configuration
The tests use pytest with the following configuration:
- **Test discovery**: Automatic discovery of `test_*.py` files
- **Coverage**: HTML and XML reports
- **Parallel execution**: Available for Python tests
- **Fail-fast**: Stop on first failure option
- **Verbose output**: Detailed test output

### Lean Test Configuration
- **Build system**: Lake package manager
- **Test execution**: Through `lake exe lean_niche`
- **Timeout**: 5 minutes per test suite
- **Error handling**: Comprehensive error reporting

## 🐛 Debugging Failed Tests

### Python Test Debugging
```bash
# Run specific test with debugging
pytest src/tests/python/test_cli.py::TestCLI::test_plot_function_command -v -s

# Run with Python debugger
pytest --pdb src/tests/python/test_visualization.py

# Run with coverage details
pytest --cov=src.python --cov-report=term-missing:skip-covered src/tests/python/
```

### LaTeX Test Debugging
```bash
# Run specific LaTeX test
pytest src/tests/latex/test_lean_to_latex.py::TestLeanToLatexConverter::test_convert_symbol -v

# Debug symbol conversion
python -c "
from src.latex.lean_to_latex import LeanToLatexConverter
conv = LeanToLatexConverter()
print('∀ test:', conv.convert_symbol('∀'))
print('Expression test:', conv.convert_expression('∀ x ∈ ℝ, x ≥ 0'))
"
```

### Lean Test Debugging
```bash
# Run specific Lean test
lake exe lean_niche  # Look for specific error messages

# Check Lean compilation
lake build src/tests/lean/test_basic.lean

# Debug with more verbose output
LEAN_DEBUG=1 lake exe lean_niche
```

## 📈 Test Metrics

### Performance Benchmarks
- **Python tests**: < 30 seconds
- **LaTeX tests**: < 10 seconds
- **Lean tests**: < 5 minutes
- **Integration tests**: < 5 seconds
- **Performance tests**: < 30 seconds

### Coverage Targets
- **Python modules**: > 90% coverage
- **LaTeX modules**: > 95% coverage
- **Lean modules**: 100% theorem verification
- **Integration**: All modules importable

## 🔄 CI/CD Integration

### GitHub Actions
```yaml
- name: Run Tests
  run: |
    python src/tests/run_tests.py --coverage --junit-xml test-results.xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Pre-commit Hooks
```yaml
- repo: local
  hooks:
    - id: run-tests
      name: Run Test Suite
      entry: python src/tests/run_tests.py --fail-fast
      language: system
      pass_filenames: false
      always_run: true
```

## 📝 Adding New Tests

### Adding Python Tests
1. Create new test file in `src/tests/python/`
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures and mocking
4. Add appropriate docstrings

### Adding LaTeX Tests
1. Add tests to `test_lean_to_latex.py`
2. Test symbol mappings
3. Test complex expressions
4. Test file conversion

### Adding Lean Tests
1. Create new `.lean` test file in `src/tests/lean/`
2. Follow naming convention: `test_*.lean`
3. Use IO Unit for test functions
4. Include comprehensive assertions

## 🚨 Troubleshooting

### Common Issues

1. **Python tests fail**: Check Python dependencies
   ```bash
   pip install -e .
   ```

2. **Lean tests fail**: Check Lean installation
   ```bash
   elan self update
   lake update
   ```

3. **Coverage reports empty**: Install coverage packages
   ```bash
   pip install pytest-cov
   ```

4. **Import errors**: Check Python path
   ```bash
   PYTHONPATH=src python -m pytest src/tests/
   ```

5. **Timeout errors**: Increase timeout for slow tests
   ```bash
   python src/tests/run_tests.py --verbose
   ```

### Getting Help

- **Test failures**: Check the detailed error output
- **Coverage issues**: Review the HTML coverage report
- **Performance problems**: Run with `--verbose` for timing information
- **Integration issues**: Check module dependencies in `lakefile.toml`

## 📚 Related Documentation

- [pytest Documentation](https://docs.pytest.org/)
- [Lean Testing Guide](https://leanprover-community.github.io/lean-testing/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Lake Package Manager](https://github.com/leanprover/lake)

---

**LeanNiche Test Suite**: Ensuring mathematical correctness through comprehensive testing.
