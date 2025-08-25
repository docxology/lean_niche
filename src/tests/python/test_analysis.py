"""Comprehensive unit tests for MathematicalAnalyzer and ComprehensiveMathematicalAnalyzer"""

import numpy as np
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.python.analysis.analysis import MathematicalAnalyzer
from src.python.analysis.comprehensive_analysis import ComprehensiveMathematicalAnalyzer


class TestMathematicalAnalyzer:
    """Comprehensive tests for MathematicalAnalyzer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = MathematicalAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization"""
        assert isinstance(self.analyzer, MathematicalAnalyzer)
        assert hasattr(self.analyzer, 'analyze_function')
        assert hasattr(self.analyzer, 'numerical_integration')
        assert hasattr(self.analyzer, 'symbolic_analysis')
        assert hasattr(self.analyzer, 'statistical_analysis')
        assert hasattr(self.analyzer, 'plot_analysis_results')

    def test_analyze_function_basic_properties(self):
        """Test basic function analysis properties"""
        def f(x):
            return x**2 - 2*x + 1

        result = self.analyzer.analyze_function(f, (-3, 3))

        assert 'mean' in result
        assert isinstance(result['mean'], float)
        assert 'zeros' in result
        assert 'extrema' in result
        assert 'asymptotes' in result
        assert 'periodicity' in result

    def test_analyze_function_linear(self):
        """Test analysis of linear function"""
        def f(x):
            return 2*x + 3

        result = self.analyzer.analyze_function(f, (-5, 5))

        assert 'mean' in result
        assert 'zeros' in result
        assert len(result['zeros']) == 1
        assert abs(result['zeros'][0] + 1.5) < 1e-6  # Zero at x = -1.5

    def test_analyze_function_trigonometric(self):
        """Test analysis of trigonometric function"""
        def f(x):
            return np.sin(x)

        result = self.analyzer.analyze_function(f, (0, 2*np.pi))

        assert 'mean' in result
        assert 'zeros' in result
        assert 'periodicity' in result
        assert result['periodicity'] is not None

    def test_analyze_function_exponential(self):
        """Test analysis of exponential function"""
        def f(x):
            return np.exp(x)

        result = self.analyzer.analyze_function(f, (-2, 2))

        assert 'mean' in result
        assert 'asymptotes' in result
        assert len(result['asymptotes']['horizontal']) == 0  # No horizontal asymptotes
        assert len(result['asymptotes']['vertical']) == 0

    def test_analyze_function_with_zeros(self):
        """Test function with multiple zeros"""
        def f(x):
            return (x - 1) * (x + 2) * (x - 3)

        result = self.analyzer.analyze_function(f, (-5, 5))

        assert 'zeros' in result
        zeros = result['zeros']
        assert len(zeros) == 3
        expected_zeros = [-2, 1, 3]
        for expected in expected_zeros:
            assert any(abs(z - expected) < 1e-6 for z in zeros)

    def test_analyze_function_with_extrema(self):
        """Test function with extrema"""
        def f(x):
            return x**3 - 3*x**2 + 2  # Has local max and min

        result = self.analyzer.analyze_function(f, (-2, 4))

        assert 'extrema' in result
        extrema = result['extrema']
        assert 'maxima' in extrema or 'minima' in extrema

    def test_numerical_integration_quad(self):
        """Test numerical integration with quad method"""
        def f(x):
            return x

        result = self.analyzer.numerical_integration(f, 0, 1, method='quad')
        assert result['method'] == 'quad'
        assert abs(result['result'] - 0.5) < 1e-6
        assert 'error' in result

    def test_numerical_integration_trapezoid(self):
        """Test numerical integration with trapezoid method"""
        def f(x):
            return x**2

        result = self.analyzer.numerical_integration(f, 0, 1, method='trapezoid')
        assert result['method'] == 'trapezoid'
        assert abs(result['result'] - 1/3) < 1e-3  # Less accurate but should be close

    def test_numerical_integration_simpson(self):
        """Test numerical integration with Simpson's rule"""
        def f(x):
            return np.sin(x)

        result = self.analyzer.numerical_integration(f, 0, np.pi, method='simpson')
        assert result['method'] == 'simpson'
        assert abs(result['result'] - 2.0) < 1e-6  # ∫sin(x) from 0 to π = 2

    def test_numerical_integration_invalid_method(self):
        """Test numerical integration with invalid method"""
        def f(x):
            return x

        with pytest.raises(ValueError):
            self.analyzer.numerical_integration(f, 0, 1, method='invalid')

    def test_symbolic_analysis_basic(self):
        """Test basic symbolic analysis"""
        expression = "x**2 + 2*x + 1"

        with patch('sympy.sympify'), patch('sympy.diff'), patch('sympy.integrate'):
            result = self.analyzer.symbolic_analysis(expression, ['x'])
            assert isinstance(result, dict)

    def test_symbolic_analysis_derivative(self):
        """Test symbolic derivative computation"""
        expression = "x**3"

        result = self.analyzer.symbolic_analysis(expression, ['x'], operation='derivative')
        assert 'derivative' in result
        assert isinstance(result['derivative'], str)

    def test_symbolic_analysis_integral(self):
        """Test symbolic integral computation"""
        expression = "x**2"

        result = self.analyzer.symbolic_analysis(expression, ['x'], operation='integral')
        assert 'integral' in result
        assert isinstance(result['integral'], str)

    def test_symbolic_analysis_simplify(self):
        """Test symbolic simplification"""
        expression = "(x**2 + 2*x + 1)/(x + 1)"

        result = self.analyzer.symbolic_analysis(expression, ['x'], operation='simplify')
        assert 'simplified' in result

    def test_symbolic_analysis_invalid_expression(self):
        """Test symbolic analysis with invalid expression"""
        result = self.analyzer.symbolic_analysis("invalid++expression", ['x'])
        assert 'error' in result or result == {}

    def test_statistical_analysis_normal(self):
        """Test statistical analysis with normal distribution"""
        data = np.random.normal(0, 1, 1000)

        result = self.analyzer.statistical_analysis(data.tolist(), alpha=0.05)

        assert 'normality_test' in result
        assert 'shapiro_wilk' in result['normality_test']
        assert 'kolmogorov_smirnov' in result['normality_test']
        assert 'distribution_fit' in result
        assert 'outliers' in result

    def test_statistical_analysis_small_sample(self):
        """Test statistical analysis with small sample"""
        data = [1, 2, 3, 4, 5]

        result = self.analyzer.statistical_analysis(data, alpha=0.05)

        assert 'mean' in result
        assert 'std' in result
        assert 'normality_test' in result

    def test_statistical_analysis_empty_data(self):
        """Test statistical analysis with empty data"""
        with pytest.raises(ValueError):
            self.analyzer.statistical_analysis([], alpha=0.05)

    def test_plot_analysis_results(self):
        """Test plotting analysis results"""
        analysis_data = {
            'mean': 5.0,
            'zeros': [-1, 2],
            'extrema': {'maxima': [(0, 10)], 'minima': [(1, -5)]}
        }

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = self.analyzer.plot_analysis_results(analysis_data, "Test Analysis", "test.png")

            mock_subplots.assert_called_once()
            mock_savefig.assert_called_once()
            assert result == mock_fig

    def test_find_zeros_basic(self):
        """Test zero finding for basic functions"""
        def f(x):
            return x**2 - 4

        zeros = self.analyzer._find_zeros(f, (-5, 5))
        assert len(zeros) == 2
        assert any(abs(z + 2) < 1e-6 for z in zeros)
        assert any(abs(z - 2) < 1e-6 for z in zeros)

    def test_find_zeros_no_zeros(self):
        """Test zero finding for function with no zeros"""
        def f(x):
            return x**2 + 1

        zeros = self.analyzer._find_zeros(f, (-2, 2))
        assert len(zeros) == 0

    def test_find_extrema_basic(self):
        """Test extrema finding"""
        def f(x):
            return x**3 - 3*x  # Has extrema at x = ±1

        extrema = self.analyzer._find_extrema(f, (-2, 2))

        assert 'maxima' in extrema or 'minima' in extrema

    def test_find_asymptotes_rational(self):
        """Test asymptote finding for rational functions"""
        def f(x):
            return 1/x

        asymptotes = self.analyzer._find_asymptotes(f, (-5, 5))

        assert 'vertical' in asymptotes
        assert len(asymptotes['vertical']) > 0

    def test_find_asymptotes_polynomial(self):
        """Test asymptote finding for polynomial functions"""
        def f(x):
            return x**2 + 3*x + 2

        asymptotes = self.analyzer._find_asymptotes(f, (-5, 5))

        assert len(asymptotes['vertical']) == 0
        assert len(asymptotes['horizontal']) == 0

    def test_analyze_periodicity_sine(self):
        """Test periodicity analysis for sine function"""
        x = np.linspace(0, 4*np.pi, 1000)
        y = np.sin(x)

        period = self.analyzer._analyze_periodicity(x, y)
        assert period is not None
        assert abs(period - 2*np.pi) < 0.1

    def test_analyze_periodicity_non_periodic(self):
        """Test periodicity analysis for non-periodic function"""
        x = np.linspace(0, 10, 100)
        y = x**2

        period = self.analyzer._analyze_periodicity(x, y)
        assert period is None


class TestComprehensiveMathematicalAnalyzer:
    """Comprehensive tests for ComprehensiveMathematicalAnalyzer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.output_dir = tempfile.mkdtemp()
        self.analyzer = ComprehensiveMathematicalAnalyzer(
            output_dir=self.output_dir,
            log_level="INFO",
            enable_logging=True
        )

    def teardown_method(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_initialization(self):
        """Test comprehensive analyzer initialization"""
        assert isinstance(self.analyzer, ComprehensiveMathematicalAnalyzer)
        assert hasattr(self.analyzer, 'comprehensive_function_analysis')
        assert hasattr(self.analyzer, 'save_analysis_results')
        assert hasattr(self.analyzer, 'create_analysis_gallery')

    def test_comprehensive_function_analysis_basic(self):
        """Test comprehensive function analysis - basic properties"""
        def f(x):
            return np.sin(x)

        results = self.analyzer.comprehensive_function_analysis(f, (0, np.pi), analysis_type='basic')

        assert 'basic' in results
        basic = results['basic']
        assert 'range' in basic
        assert 'zeros' in basic
        assert 'mean' in basic

    def test_comprehensive_function_analysis_calculus(self):
        """Test comprehensive function analysis - calculus properties"""
        def f(x):
            return x**3 - 3*x**2 + 2

        results = self.analyzer.comprehensive_function_analysis(f, (-2, 4), analysis_type='calculus')

        assert 'calculus' in results
        calculus = results['calculus']
        assert 'derivative' in calculus
        assert 'second_derivative' in calculus
        assert 'critical_points' in calculus

    def test_comprehensive_function_analysis_statistical(self):
        """Test comprehensive function analysis - statistical properties"""
        def f(x):
            return np.random.normal(0, 1, 1)[0] + x  # Linear with noise

        results = self.analyzer.comprehensive_function_analysis(f, (-5, 5), analysis_type='statistical')

        assert 'statistical' in results
        statistical = results['statistical']
        assert 'distribution_fit' in statistical
        assert 'complexity_measures' in statistical

    def test_comprehensive_function_analysis_advanced(self):
        """Test comprehensive function analysis - advanced properties"""
        def f(x):
            return np.sin(x) * np.exp(-x/10)  # Damped sine wave

        results = self.analyzer.comprehensive_function_analysis(f, (0, 20), analysis_type='advanced')

        assert 'advanced' in results
        advanced = results['advanced']
        assert 'lyapunov_exponent' in advanced
        assert 'chaos_indicators' in advanced

    def test_comprehensive_function_analysis_full(self):
        """Test comprehensive function analysis - full analysis"""
        def f(x):
            return np.sin(x) + 0.5*np.sin(3*x)  # Harmonic series

        results = self.analyzer.comprehensive_function_analysis(f, (0, 4*np.pi), analysis_type='full')

        assert 'basic' in results
        assert 'calculus' in results
        assert 'statistical' in results
        assert 'advanced' in results

    def test_save_analysis_results(self):
        """Test saving analysis results"""
        results = {
            'basic': {'mean': 5.0, 'zeros': [0]},
            'calculus': {'critical_points': []}
        }

        output_path = self.analyzer.save_analysis_results(results, filename="test_results")

        assert output_path.exists()
        assert output_path.suffix == '.json'

        # Verify content
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
            assert 'basic' in saved_data
            assert saved_data['basic']['mean'] == 5.0

    def test_create_analysis_gallery(self):
        """Test creating analysis gallery"""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            results = self.analyzer.create_analysis_gallery()

            assert isinstance(results, dict)
            assert 'functions_analyzed' in results
            assert 'plots_created' in results

            # Verify multiple savefig calls were made
            assert mock_savefig.call_count >= 3  # At least 3 plots

    def test_find_zeros_comprehensive(self):
        """Test comprehensive zero finding"""
        def f(x):
            return (x - 1) * (x + 1) * (x - 2)

        zeros = self.analyzer._find_zeros_comprehensive(f, (-5, 5))

        assert len(zeros) >= 3  # Should find all three zeros
        expected_zeros = [-1, 1, 2]
        for expected in expected_zeros:
            assert any(abs(z - expected) < 1e-2 for z in zeros)

    def test_analyze_monotonicity_increasing(self):
        """Test monotonicity analysis for increasing function"""
        x = np.linspace(0, 10, 100)
        y = x**2  # Strictly increasing

        result = self.analyzer._analyze_monotonicity(x, y)

        assert 'overall_trend' in result
        assert result['overall_trend'] in ['increasing', 'strictly_increasing']

    def test_analyze_boundedness_bounded(self):
        """Test boundedness analysis for bounded function"""
        y = np.sin(np.linspace(0, 10, 100))

        result = self.analyzer._analyze_boundedness(y)

        assert 'bounded_above' in result
        assert 'bounded_below' in result
        assert result['bounded_above'] is True
        assert result['bounded_below'] is True

    def test_analyze_symmetry_even(self):
        """Test symmetry analysis for even function"""
        def f(x):
            return x**2

        result = self.analyzer._analyze_symmetry(f, (-5, 5))

        assert 'even_function' in result

    def test_analyze_symmetry_odd(self):
        """Test symmetry analysis for odd function"""
        def f(x):
            return x**3

        result = self.analyzer._analyze_symmetry(f, (-5, 5))

        assert 'odd_function' in result

    def test_find_extrema_numerical(self):
        """Test numerical extrema finding"""
        def f(x):
            return x**4 - 4*x**2 + 3  # Has two minima

        extrema = self.analyzer._find_extrema_numerical(f, (-3, 3))

        assert len(extrema) >= 2  # Should find local minima

    def test_analyze_concavity_convex(self):
        """Test concavity analysis for convex function"""
        def f(x):
            return x**2

        result = self.analyzer._analyze_concavity(f, (-2, 2))

        assert 'convex_regions' in result

    def test_analyze_distribution_fit_normal(self):
        """Test distribution fitting for normal-like data"""
        data = np.random.normal(0, 1, 1000)

        result = self.analyzer._analyze_distribution_fit(data)

        assert 'best_fit' in result
        assert 'parameters' in result

    def test_compute_complexity_measures(self):
        """Test complexity measures computation"""
        data = np.random.rand(1000)

        result = self.analyzer._compute_complexity_measures(data)

        assert 'approximate_entropy' in result
        assert 'sample_entropy' in result
        assert 'fractal_dimension' in result

    def test_analyze_chaos_indicators_chaotic(self):
        """Test chaos indicators for potentially chaotic system"""
        def f(x):
            return 4*x*(1-x)  # Logistic map at r=4 (chaotic)

        result = self.analyzer._analyze_chaos_indicators(f, (0, 1))

        assert 'lyapunov_exponent' in result

    def test_analyze_bifurcation_simple(self):
        """Test simple bifurcation analysis"""
        def f(x):
            return 2.5*x*(1-x)  # Logistic map at r=2.5

        result = self.analyzer._analyze_bifurcation_simple(f, (0, 1))

        assert 'fixed_points' in result

    def test_analyze_sensitivity_dependent(self):
        """Test sensitivity analysis"""
        def f(x):
            return 4*x*(1-x)  # Sensitive dependence

        result = self.analyzer._analyze_sensitivity(f, (0, 1))

        assert 'sensitivity_measure' in result

    def test_compute_information_measures(self):
        """Test information measures computation"""
        data = np.random.rand(1000)

        result = self.analyzer._compute_information_measures(data)

        assert 'entropy' in result
        assert 'mutual_information' in result

    def test_create_analysis_visualizations(self):
        """Test analysis visualization creation"""
        def f(x):
            return np.sin(x)

        results = {'basic': {'zeros': [0, np.pi], 'mean': 0}}

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            visualizations = self.analyzer._create_analysis_visualizations(f, (0, 2*np.pi), results)

            assert 'function_plot' in visualizations
            assert 'derivative_plot' in visualizations
            assert 'analysis_summary' in visualizations

    def test_generate_comprehensive_report(self):
        """Test comprehensive report generation"""
        def f(x):
            return x**2

        results = {
            'basic': {'mean': 5.0, 'zeros': [0]},
            'calculus': {'critical_points': [0]},
            'statistical': {'distribution_fit': {'normal': {'mean': 5.0}}}
        }

        report = self.analyzer._generate_comprehensive_report(f, (-5, 5), results)

        assert isinstance(report, str)
        assert len(report) > 100  # Should be a substantial report
        assert 'Function Analysis Report' in report
        assert 'Basic Properties' in report

    def test_error_handling_invalid_function(self):
        """Test error handling for invalid function"""
        def invalid_f(x):
            return x / 0  # Will cause division by zero

        with pytest.raises(Exception):
            self.analyzer.comprehensive_function_analysis(invalid_f, (-1, 1))

    def test_error_handling_invalid_domain(self):
        """Test error handling for invalid domain"""
        def f(x):
            return x**2

        with pytest.raises(ValueError):
            self.analyzer.comprehensive_function_analysis(f, (5, -5))  # Invalid domain

    def test_logging_integration(self):
        """Test logging integration"""
        def f(x):
            return x**2

        with patch.object(self.analyzer.logger, 'info') as mock_info:
            self.analyzer.comprehensive_function_analysis(f, (-1, 1))
            mock_info.assert_called()  # Should log operation


