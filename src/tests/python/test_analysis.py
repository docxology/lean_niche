"""Unit tests for MathematicalAnalyzer and ComprehensiveMathematicalAnalyzer"""

import numpy as np
from src.python.analysis.analysis import MathematicalAnalyzer
from src.python.analysis.comprehensive_analysis import ComprehensiveMathematicalAnalyzer


def test_analyze_function_basic_properties(tmp_path):
    analyzer = MathematicalAnalyzer()

    def f(x):
        return x**2 - 2*x + 1

    res = analyzer.analyze_function(f, (-3, 3))

    assert 'mean' in res
    assert isinstance(res['mean'], float)
    assert 'zeros' in res


def test_numerical_integration_quad():
    analyzer = MathematicalAnalyzer()

    def f(x):
        return x

    r = analyzer.numerical_integration(f, 0, 1, method='quad')
    assert r['method'] == 'quad'
    assert abs(r['result'] - 0.5) < 1e-6


def test_comprehensive_analyzer_basic(tmp_path):
    analyzer = ComprehensiveMathematicalAnalyzer(output_dir=str(tmp_path))

    def f(x):
        return np.sin(x)

    results = analyzer.comprehensive_function_analysis(f, (0, np.pi), analysis_type='basic')

    assert 'basic' in results
    basic = results['basic']
    assert 'range' in basic
    assert 'zeros' in basic


