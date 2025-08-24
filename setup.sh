#!/bin/bash

# LeanNiche Comprehensive Setup Script
# Complete installation and configuration for the LeanNiche mathematical research environment

set -e  # Exit on any error

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

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to check system requirements
check_system_requirements() {
    print_header "System Requirements Check"

    local os=$(detect_os)
    print_status "Detected OS: $os"

    # Check available memory
    if command_exists free; then
        local mem_gb=$(free -m | awk 'NR==2{printf "%.1f", $2/1024}')
        print_status "Available memory: ${mem_gb}GB"
        if (( $(echo "$mem_gb < 4" | bc -l) )); then
            print_warning "Less than 4GB RAM detected. Lean compilation may be slow."
        fi
    fi

    # Check available disk space
    if command_exists df; then
        local disk_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
        print_status "Available disk space: ${disk_gb}GB"
        if [ "$disk_gb" -lt 10 ]; then
            print_warning "Less than 10GB disk space available. Consider freeing up space."
        fi
    fi

    print_success "System requirements check completed"
}

# Function to setup Elan (Lean version manager)
setup_elan() {
    print_header "Setting up Elan (Lean Version Manager)"

    if command_exists elan; then
        print_success "Elan is already installed"
        elan --version
    else
        print_status "Installing Elan..."

        if command_exists curl; then
            curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
            source $HOME/.elan/env
        elif command_exists wget; then
            wget -qO- https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | sh -s -- -y
            source $HOME/.elan/env
        else
            print_error "Neither curl nor wget found. Please install curl or wget and try again."
            exit 1
        fi

        if command_exists elan; then
            print_success "Elan installed successfully"
            elan --version
        else
            print_error "Failed to install Elan"
            exit 1
        fi
    fi
}

# Function to setup Lean toolchain
setup_lean() {
    print_header "Setting up Lean Toolchain"

    if command_exists lean; then
        print_success "Lean is available"
        lean --version
    else
        print_error "Lean not found. Please ensure Elan is properly installed and run: source \$HOME/.elan/env"
        exit 1
    fi

    # Check Lake (Lean's package manager)
    if command_exists lake; then
        print_success "Lake is available"
        lake --version
    else
        print_error "Lake not found. Please update Elan: elan self update"
        exit 1
    fi

    # Verify toolchain
    if [ -f "lean-toolchain" ]; then
        local toolchain=$(cat lean-toolchain)
        print_status "Using Lean toolchain: $toolchain"
        elan override set $toolchain
    else
        print_warning "No lean-toolchain file found. Using default Lean version."
    fi
}

# Function to setup Python environment
setup_python() {
    print_header "Setting up Python Environment"

    if ! command_exists python3; then
        print_error "Python 3 not found. Please install Python 3.8+ and try again."
        exit 1
    fi

    local python_version=$(python3 --version)
    print_status "Python version: $python_version"

    # Check if we should use virtual environment
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    fi

    print_status "Activating virtual environment..."
    source venv/bin/activate

    print_status "Installing Python dependencies..."
    if [ -f "pyproject.toml" ]; then
        if command_exists pip; then
            pip install --upgrade pip
            pip install -e .
            print_success "Python dependencies installed"
        else
            print_error "pip not found. Please install pip and try again."
            exit 1
        fi
    else
        print_warning "No pyproject.toml found. Installing basic dependencies..."
        pip install matplotlib numpy seaborn plotly pandas scipy networkx rich click
        print_success "Basic Python dependencies installed"
    fi
}

# Function to create Lean configuration files
create_lean_config() {
    print_header "Creating Lean Configuration Files"

    # Create lakefile.toml if it doesn't exist
    if [ ! -f "lakefile.toml" ]; then
        print_status "Creating lakefile.toml..."
        cat > lakefile.toml << 'EOF'
-- LeanNiche Project Configuration
name = "LeanNiche"
version = "0.1.0"
description = "Advanced Deep Research Environment for Mathematics"

[[require]]
name = "mathlib"
version = "4.22.0"

[[lean_lib]]
name = "LeanNiche"
srcDir = "src/lean"

[[lean_exe]]
name = "lean_niche"
srcDir = "src/lean"

[[lean_exe]]
name = "test_suite"
srcDir = "src/tests"

[buildOptions]
-- Additional build options can be added here
EOF
        print_success "lakefile.toml created"
    else
        print_status "lakefile.toml already exists"
    fi
}

# Function to create Python configuration
create_python_config() {
    print_header "Creating Python Configuration"

    # Update or create pyproject.toml
    if [ ! -f "pyproject.toml" ]; then
        print_status "Creating pyproject.toml..."
        cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=45", "wheel", "setuptools-scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "lean-niche"
version = "0.1.0"
description = "Python utilities for LeanNiche mathematical research environment"
readme = "README.md"
authors = [
    {name = "LeanNiche Team"}
]
license = {file = "LICENSE"}
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Mathematics",
]
requires-python = ">=3.8"
dependencies = [
    "matplotlib>=3.5.0",
    "numpy>=1.21.0",
    "seaborn>=0.11.0",
    "plotly>=5.0.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "networkx>=2.6.0",
    "rich>=10.0.0",
    "click>=8.0.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]

[project.scripts]
lean-niche = "src.python.cli:main"

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "black>=21.0.0",
    "isort>=5.0.0",
    "mypy>=0.900",
    "flake8>=4.0.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"*" = ["*.txt", "*.md"]
EOF
        print_success "pyproject.toml created"
    else
        print_status "pyproject.toml already exists"
    fi
}

# Function to setup directory structure
setup_directories() {
    print_header "Setting up Directory Structure"

    local directories=(
        "build"
        "visualizations"
        "data"
        "notebooks"
        "docs/api"
        "logs"
    )

    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory exists: $dir"
        fi
    done

    print_success "Directory structure setup completed"
}

# Function to build Lean project
build_lean_project() {
    print_header "Building Lean Project"

    print_status "Updating dependencies..."
    lake update

    print_status "Building project..."
    if lake build; then
        print_success "Lean project built successfully"
    else
        print_error "Failed to build Lean project"
        exit 1
    fi

    print_status "Running tests..."
    if lake exe test_suite; then
        print_success "All tests passed"
    else
        print_warning "Some tests failed. Check output above."
    fi
}

# Function to create configuration files
create_config_files() {
    print_header "Creating Configuration Files"

    # Create .gitignore if it doesn't exist
    if [ ! -f ".gitignore" ]; then
        print_status "Creating .gitignore..."
        cat > .gitignore << 'EOF'
# Build artifacts
build/
*.olean
*.c
*.o

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
.venv

# Visualizations and data
visualizations/
*.png
*.jpg
*.svg
*.pdf
data/
*.json
*.csv

# Logs
logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db
EOF
        print_success ".gitignore created"
    fi

    # Create .env.example
    if [ ! -f ".env.example" ]; then
        print_status "Creating .env.example..."
        cat > .env.example << 'EOF'
# LeanNiche Environment Configuration

# Lean settings
LEAN_VERSION=4.22.0
MATHLIB_VERSION=latest

# Python settings
PYTHONPATH=src/python

# Visualization settings
VISUALIZATION_BACKEND=matplotlib
PLOT_DPI=300
SAVE_FORMAT=png

# Performance settings
MAX_MEMORY_GB=8
MAX_THREADS=4
TIMEOUT_SECONDS=300

# Development settings
DEBUG=false
VERBOSE=true
EOF
        print_success ".env.example created"
    fi
}

# Function to create documentation
create_documentation() {
    print_header "Setting up Documentation"

    # Create basic Jupyter notebook example
    if [ ! -f "notebooks/LeanNiche_Examples.ipynb" ]; then
        print_status "Creating example Jupyter notebook..."
        cat > notebooks/LeanNiche_Examples.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LeanNiche Mathematical Research Environment\n",
    "\n",
    "This notebook demonstrates the LeanNiche environment capabilities.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import LeanNiche Python utilities\n",
    "import sys\n",
    "sys.path.append('src/python')\n",
    "\n",
    "from lean_niche.visualization import MathematicalVisualizer\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "print(\"LeanNiche Python utilities loaded successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Example: Plot a mathematical function\n",
    "viz = MathematicalVisualizer()\n",
    "\n",
    "def f(x):\n",
    "    return np.sin(x) * np.exp(-x/10)\n",
    "\n",
    "fig = viz.plot_function(f, (0, 20), \"Damped Sine Wave\")\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF
        print_success "Example notebook created"
    fi
}

# Function to display usage instructions
display_usage() {
    print_header "LeanNiche Setup Complete!"

    echo -e "${CYAN}🚀 Getting Started:${NC}"
    echo ""
    echo "1. Activate the environment:"
    echo -e "${YELLOW}   source venv/bin/activate${NC}"
    echo ""
    echo "2. Run Lean examples:"
    echo -e "${YELLOW}   lake exe lean_niche${NC}"
    echo ""
    echo "3. Use Python utilities:"
    echo -e "${YELLOW}   python -m src.python.cli --help${NC}"
    echo ""
    echo "4. Run tests:"
    echo -e "${YELLOW}   ./scripts/test.sh${NC}"
    echo ""
    echo "5. Build and analyze:"
    echo -e "${YELLOW}   ./scripts/build.sh${NC}"
    echo -e "${YELLOW}   ./scripts/analyze.sh${NC}"
    echo ""
    echo "6. Create visualizations:"
    echo -e "${YELLOW}   python -m src.python.cli gallery${NC}"
    echo ""
    echo "7. Start Jupyter:"
    echo -e "${YELLOW}   jupyter notebook notebooks/${NC}"
    echo ""
    echo -e "${CYAN}📚 Available Modules:${NC}"
    echo "  • Basic Mathematics"
    echo "  • Advanced Mathematics"
    echo "  • Set Theory & Topology"
    echo "  • Statistics & Probability"
    echo "  • Dynamical Systems"
    echo "  • Lyapunov Theory"
    echo "  • Computational Methods"
    echo ""
    echo -e "${CYAN}🛠️  Development Tools:${NC}"
    echo "  • Comprehensive test suite"
    echo "  • Code analysis tools"
    echo "  • Visualization gallery"
    echo "  • Documentation generation"
    echo ""
    echo -e "${GREEN}🎉 LeanNiche is ready for mathematical research!${NC}"
}

# Main setup function
main() {
    print_header "LeanNiche Comprehensive Setup"
    echo -e "${CYAN}Setting up complete mathematical research environment...${NC}"
    echo ""

    # Make scripts executable
    chmod +x scripts/*.sh

    # Run setup steps
    check_system_requirements
    setup_elan
    setup_lean
    setup_python
    create_lean_config
    create_python_config
    setup_directories
    create_config_files
    create_documentation
    build_lean_project

    display_usage
}

# Run main function with all arguments
main "$@"
