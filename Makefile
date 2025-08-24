# LeanNiche Makefile
# Common commands for LeanNiche development and usage

.PHONY: help setup build test clean docs analyze viz install-deps update-deps

# Default target
help:
	@echo "LeanNiche - Deep Research Environment"
	@echo ""
	@echo "Available commands:"
	@echo "  setup         - Run comprehensive setup script"
	@echo "  build         - Build all Lean modules"
	@echo "  test          - Run comprehensive test suite"
	@echo "  clean         - Clean build artifacts"
	@echo "  docs          - Generate documentation"
	@echo "  analyze       - Analyze code structure and metrics"
	@echo "  viz           - Create visualization gallery"
	@echo "  install-deps  - Install all dependencies"
	@echo "  update-deps   - Update all dependencies"
	@echo "  run           - Run main Lean environment"
	@echo "  notebook      - Start Jupyter notebook"
	@echo "  format        - Format code"
	@echo "  lint          - Run linting checks"
	@echo ""

# Setup
setup:
	./setup.sh

# Build
build:
	lake build

# Test
test:
	./scripts/test.sh

# Clean
clean:
	lake clean
	rm -rf build/ visualizations/ __pycache__/ *.log

# Documentation
docs:
	./scripts/docs.sh

# Analysis
analyze:
	./scripts/analyze.sh

# Visualization
viz:
	source venv/bin/activate && lean-niche-viz

# Install dependencies
install-deps:
	lake update
	source venv/bin/activate && uv sync

# Update dependencies
update-deps:
	lake update
	source venv/bin/activate && uv sync

# Run main environment
run:
	lake exe lean_niche

# Jupyter notebook
notebook:
	source venv/bin/activate && jupyter notebook notebooks/

# Format code
format:
	source venv/bin/activate && black src/python/
	source venv/bin/activate && isort src/python/

# Lint code
lint:
	source venv/bin/activate && flake8 src/python/
	source venv/bin/activate && mypy src/python/

# Quick development cycle
dev: clean build test

# Production build
prod: clean
	lake build --release

# Install for development
install-dev: install-deps
	source venv/bin/activate && uv pip install -e .[dev]

# Create distribution
dist:
	source venv/bin/activate && python -m build

# Publish to PyPI (use with caution)
publish:
	source venv/bin/activate && python -m twine upload dist/*

# Docker support (if needed in future)
docker-build:
	docker build -t lean-niche .

docker-run:
	docker run -it lean-niche

# Performance profiling
profile:
	lake build --profile
	source venv/bin/activate && python -m cProfile -s time src/python/cli.py --help

# Memory usage check
memory-check:
	@echo "Lean build memory usage:"
	/usr/bin/time -v lake build 2>&1 | grep -E "(Maximum resident|User time|System time)"
	@echo ""
	@echo "Python memory usage:"
	source venv/bin/activate && python -c "import psutil; import os; process = psutil.Process(os.getpid()); print(f'Current memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')"

# Help for specific modules
help-basic:
	@echo "Basic.lean - Fundamental mathematical proofs"
	@echo "  • Natural number arithmetic"
	@echo "  • Basic algebraic properties"
	@echo "  • Simple induction proofs"

help-advanced:
	@echo "Advanced.lean - Advanced mathematical concepts"
	@echo "  • Prime number theory"
	@echo "  • Number sequences"
	@echo "  • Infinite descent proofs"

help-dynamical:
	@echo "DynamicalSystems.lean - Dynamical systems theory"
	@echo "  • State spaces and flow functions"
	@echo "  • Stability definitions"
	@echo "  • Poincaré-Bendixson theorem"

help-lyapunov:
	@echo "Lyapunov.lean - Stability analysis"
	@echo "  • Lyapunov functions"
	@echo "  • Stability theorems"
	@echo "  • Control theory applications"
