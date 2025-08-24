"""LeanNiche Python Utilities

Python utilities and visualization tools for the LeanNiche mathematical research environment.
"""

__version__ = "0.1.0"
__author__ = "LeanNiche Team"
__description__ = "Python utilities for mathematical visualization and analysis"

# Import all major components for easy access
from .core.lean_runner import LeanRunner
from .core.orchestrator_base import LeanNicheOrchestratorBase
from .analysis.comprehensive_analysis import ComprehensiveMathematicalAnalyzer
from .analysis.data_generator import MathematicalDataGenerator
from .visualization.visualization import MathematicalVisualizer
from .utils.cli import cli

__all__ = [
    'LeanRunner',
    'LeanNicheOrchestratorBase',
    'ComprehensiveMathematicalAnalyzer',
    'MathematicalDataGenerator',
    'MathematicalVisualizer',
    'cli'
]
