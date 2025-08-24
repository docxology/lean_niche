#!/usr/bin/env python3
"""
Comprehensive Mathematical Analysis Tools

Advanced mathematical analysis, verification, and computational tools
for the LeanNiche environment with full logging and output capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats, optimize, integrate, special, signal
from typing import List, Dict, Any, Tuple, Callable, Optional, Union
import logging
from pathlib import Path
import json
from datetime import datetime
import sympy as sp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress
import warnings
import traceback

console = Console()


class ComprehensiveMathematicalAnalyzer:
    """Comprehensive mathematical analysis and verification tools"""

    def __init__(self, output_dir: str = "comprehensive_analysis",
                 log_level: str = "INFO", enable_logging: bool = True):
        """Initialize the comprehensive analyzer"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        self.setup_logging(log_level, enable_logging)

        # Setup matplotlib style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # Store analysis results
        self.analysis_history = []

        self.logger.info(f"ComprehensiveMathematicalAnalyzer initialized with output_dir: {self.output_dir}")

    def setup_logging(self, log_level: str, enable_logging: bool):
        """Setup comprehensive logging"""
        self.logger = logging.getLogger(__name__)

        if not enable_logging:
            self.logger.setLevel(logging.CRITICAL + 1)
            return

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler
        log_file = self.output_dir / "comprehensive_analysis.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, log_level))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, log_level))

        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(getattr(logging, log_level))

    def log_operation(self, operation: str, inputs: Dict[str, Any],
                     outputs: Dict[str, Any], success: bool = True,
                     error: Optional[str] = None):
        """Log an operation with comprehensive information"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'inputs': inputs,
            'outputs': outputs,
            'success': success,
            'error': error
        }

        self.analysis_history.append(log_entry)

        if success:
            self.logger.info(f"✅ {operation} completed successfully")
            self.logger.debug(f"   Inputs: {inputs}")
            self.logger.debug(f"   Outputs: {outputs}")
        else:
            self.logger.error(f"❌ {operation} failed: {error}")

    def comprehensive_function_analysis(self, func: Callable[[float], float],
                                     domain: Tuple[float, float],
                                     analysis_type: str = "full") -> Dict[str, Any]:
        """Comprehensive function analysis with full logging"""
        self.logger.info(f"Starting comprehensive function analysis for {analysis_type} analysis")

        inputs = {'domain': domain, 'analysis_type': analysis_type}

        try:
            results = {}

            # Basic analysis
            if analysis_type in ["full", "basic"]:
                results['basic'] = self._analyze_basic_properties(func, domain)

            # Calculus analysis
            if analysis_type in ["full", "calculus"]:
                results['calculus'] = self._analyze_calculus_properties(func, domain)

            # Statistical analysis
            if analysis_type in ["full", "statistical"]:
                results['statistical'] = self._analyze_statistical_properties(func, domain)

            # Advanced analysis
            if analysis_type in ["full", "advanced"]:
                results['advanced'] = self._analyze_advanced_properties(func, domain)

            # Create visualizations
            self._create_analysis_visualizations(func, domain, results)

            # Generate report
            report = self._generate_comprehensive_report(func, domain, results)

            outputs = {'results': results, 'report': report}
            self.log_operation("comprehensive_function_analysis", inputs, outputs, True)

            return outputs

        except Exception as e:
            error_msg = f"Function analysis failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            self.log_operation("comprehensive_function_analysis", inputs, {}, False, error_msg)
            raise

    def _analyze_basic_properties(self, func: Callable[[float], float],
                                domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze basic function properties"""
        x = np.linspace(domain[0], domain[1], 1000)
        y = func(x)

        basic_props = {
            'domain': domain,
            'range': (float(np.min(y)), float(np.max(y))),
            'zeros': self._find_zeros_comprehensive(func, domain),
            'monotonicity': self._analyze_monotonicity(x, y),
            'boundedness': self._analyze_boundedness(y),
            'periodicity': self._analyze_periodicity_advanced(x, y),
            'symmetry': self._analyze_symmetry(func, domain)
        }

        self.logger.info(f"Basic properties analyzed: range={basic_props['range']}")
        return basic_props

    def _analyze_calculus_properties(self, func: Callable[[float], float],
                                   domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze calculus properties"""
        try:
            # Numerical differentiation
            h = 1e-6
            x0 = (domain[0] + domain[1]) / 2

            # Central difference for derivative
            derivative = (func(x0 + h) - func(x0 - h)) / (2 * h)

            # Numerical integration
            integral, error = integrate.quad(func, domain[0], domain[1])

            # Find extrema
            extrema = self._find_extrema_numerical(func, domain)

            calculus_props = {
                'derivative_at_center': float(derivative),
                'integral_over_domain': float(integral),
                'integral_error': float(error),
                'local_extrema': extrema,
                'concavity_analysis': self._analyze_concavity(func, domain)
            }

            self.logger.info(f"Calculus properties: integral={calculus_props['integral_over_domain']:.4f}")
            return calculus_props

        except Exception as e:
            self.logger.warning(f"Calculus analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_statistical_properties(self, func: Callable[[float], float],
                                      domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze statistical properties"""
        x = np.linspace(domain[0], domain[1], 1000)
        y = func(x)

        statistical_props = {
            'mean': float(np.mean(y)),
            'median': float(np.median(y)),
            'std': float(np.std(y)),
            'var': float(np.var(y)),
            'skewness': float(stats.skew(y)),
            'kurtosis': float(stats.kurtosis(y)),
            'percentiles': {
                '25': float(np.percentile(y, 25)),
                '50': float(np.percentile(y, 50)),
                '75': float(np.percentile(y, 75))
            },
            'distribution_fit': self._analyze_distribution_fit(y)
        }

        self.logger.info(f"Statistical properties: mean={statistical_props['mean']:.4f}, std={statistical_props['std']:.4f}")
        return statistical_props

    def _analyze_advanced_properties(self, func: Callable[[float], float],
                                   domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze advanced mathematical properties"""
        x = np.linspace(domain[0], domain[1], 1000)
        y = func(x)

        advanced_props = {
            'fractal_dimension': self._estimate_fractal_dimension(x, y),
            'lyapunov_exponent': self._estimate_lyapunov_exponent(func, domain),
            'complexity_measures': self._compute_complexity_measures(y),
            'chaos_indicators': self._analyze_chaos_indicators(func, domain),
            'information_theory': self._compute_information_measures(y)
        }

        self.logger.info(f"Advanced properties analyzed")
        return advanced_props

    def _find_zeros_comprehensive(self, func: Callable[[float], float],
                                domain: Tuple[float, float]) -> List[float]:
        """Comprehensive zero finding"""
        zeros = []

        try:
            # Use scipy's root finding
            result = optimize.root_scalar(func, bracket=domain, method='brentq')
            if result.converged:
                zeros.append(float(result.root))

            # Also check for zeros using numpy
            x = np.linspace(domain[0], domain[1], 10000)
            y = func(x)

            # Find sign changes
            sign_changes = []
            for i in range(1, len(y)):
                if np.sign(y[i-1]) != np.sign(y[i]) and y[i-1] * y[i] <= 0:
                    # Refine the zero
                    try:
                        zero = optimize.brentq(func, x[i-1], x[i])
                        if domain[0] <= zero <= domain[1]:
                            sign_changes.append(float(zero))
                    except ValueError:
                        pass

            zeros.extend(sign_changes)
            zeros = list(set(zeros))  # Remove duplicates
            zeros.sort()

        except Exception as e:
            self.logger.warning(f"Zero finding failed: {e}")

        return zeros

    def _analyze_monotonicity(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze function monotonicity"""
        if len(y) < 3:
            return {'type': 'unknown', 'confidence': 0}

        # Compute differences
        dy = np.diff(y)
        dx = np.diff(x)

        # Analyze trends
        positive_derivatives = np.sum(dy > 0)
        negative_derivatives = np.sum(dy < 0)
        total_points = len(dy)

        if positive_derivatives / total_points > 0.8:
            return {'type': 'increasing', 'confidence': positive_derivatives / total_points}
        elif negative_derivatives / total_points > 0.8:
            return {'type': 'decreasing', 'confidence': negative_derivatives / total_points}
        else:
            return {'type': 'non_monotonic', 'confidence': 0}

    def _analyze_boundedness(self, y: np.ndarray) -> Dict[str, Any]:
        """Analyze function boundedness"""
        return {
            'bounded': not (np.isinf(y).any() or np.isnan(y).any()),
            'lower_bound': float(np.min(y)) if np.isfinite(y).all() else None,
            'upper_bound': float(np.max(y)) if np.isfinite(y).all() else None
        }

    def _analyze_periodicity_advanced(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Advanced periodicity analysis"""
        if len(y) < 20:
            return None

        # Compute autocorrelation
        correlation = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
        correlation = correlation[correlation.size // 2:]

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(correlation) - 1):
            if correlation[i] > correlation[i-1] and correlation[i] > correlation[i+1]:
                if correlation[i] > 0.3 * correlation[0]:  # Significant peak
                    peaks.append(i)

        if peaks:
            # Estimate period as first significant peak
            period_idx = peaks[0]
            period = abs(x[period_idx] - x[0])
            return float(period)

        return None

    def _analyze_symmetry(self, func: Callable[[float], float],
                         domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze function symmetry"""
        center = (domain[0] + domain[1]) / 2

        # Check even symmetry: f(x) = f(-x)
        even_score = 0
        # Check odd symmetry: f(x) = -f(-x)
        odd_score = 0

        test_points = np.linspace(domain[0], domain[1], 100)
        for x in test_points:
            if domain[0] <= -x <= domain[1]:  # Check if -x is in domain
                try:
                    f_x = func(x)
                    f_neg_x = func(-x)

                    # Even symmetry check
                    if abs(f_x - f_neg_x) < 1e-6:
                        even_score += 1

                    # Odd symmetry check
                    if abs(f_x + f_neg_x) < 1e-6:
                        odd_score += 1

                except:
                    pass

        total_tests = len(test_points)
        even_ratio = even_score / total_tests
        odd_ratio = odd_score / total_tests

        if even_ratio > 0.8:
            return {'type': 'even', 'confidence': even_ratio}
        elif odd_ratio > 0.8:
            return {'type': 'odd', 'confidence': odd_ratio}
        else:
            return {'type': 'neither', 'confidence': max(even_ratio, odd_ratio)}

    def _find_extrema_numerical(self, func: Callable[[float], float],
                              domain: Tuple[float, float]) -> List[Dict[str, Any]]:
        """Find extrema using numerical optimization"""
        extrema = []

        try:
            # Find minimum
            min_result = optimize.minimize_scalar(func, bounds=domain, method='bounded')
            if min_result.success:
                extrema.append({
                    'type': 'minimum',
                    'x': float(min_result.x),
                    'y': float(min_result.fun)
                })

            # Find maximum
            max_result = optimize.minimize_scalar(lambda x: -func(x), bounds=domain, method='bounded')
            if max_result.success:
                extrema.append({
                    'type': 'maximum',
                    'x': float(max_result.x),
                    'y': float(-max_result.fun)
                })

        except Exception as e:
            self.logger.warning(f"Extrema finding failed: {e}")

        return extrema

    def _analyze_concavity(self, func: Callable[[float], float],
                          domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze function concavity"""
        # Sample points for second derivative analysis
        x = np.linspace(domain[0], domain[1], 100)
        h = 1e-6

        # Numerical second derivative
        second_derivatives = []
        for xi in x:
            try:
                f_prime = (func(xi + h) - func(xi - h)) / (2 * h)
                f_prime_plus = (func(xi + h + h) - func(xi + h - h)) / (2 * h)
                f_second = (f_prime_plus - f_prime) / h
                second_derivatives.append(f_second)
            except:
                second_derivatives.append(0)

        second_derivatives = np.array(second_derivatives)

        # Analyze concavity
        convex_points = np.sum(second_derivatives > 0)
        concave_points = np.sum(second_derivatives < 0)
        total_points = len(second_derivatives)

        convex_ratio = convex_points / total_points
        concave_ratio = concave_points / total_points

        if convex_ratio > 0.8:
            return {'type': 'convex', 'confidence': convex_ratio}
        elif concave_ratio > 0.8:
            return {'type': 'concave', 'confidence': concave_ratio}
        else:
            return {'type': 'mixed', 'confidence': max(convex_ratio, concave_ratio)}

    def _analyze_distribution_fit(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze how well data fits various distributions"""
        distributions = {
            'normal': stats.norm,
            'exponential': stats.expon,
            'gamma': stats.gamma,
            'beta': stats.beta,
            'uniform': stats.uniform
        }

        fits = {}

        for name, distribution in distributions.items():
            try:
                # Fit distribution
                params = distribution.fit(data)
                # Compute KS test
                ks_stat, ks_p_value = stats.kstest(data, distribution.cdf, args=params)

                fits[name] = {
                    'parameters': params,
                    'ks_statistic': float(ks_stat),
                    'ks_p_value': float(ks_p_value),
                    'goodness_of_fit': float(1 - ks_stat)  # Higher is better
                }

            except Exception as e:
                fits[name] = {'error': str(e)}

        # Find best fit
        valid_fits = {k: v for k, v in fits.items() if 'error' not in v}
        if valid_fits:
            best_fit = max(valid_fits.items(), key=lambda x: x[1]['goodness_of_fit'])
            fits['best_fit'] = best_fit[0]

        return fits

    def _estimate_fractal_dimension(self, x: np.ndarray, y: np.ndarray) -> float:
        """Estimate fractal dimension using box counting method"""
        try:
            # Simple box counting for 2D curve
            # This is a simplified implementation
            n_boxes = len(np.unique(np.round(y * 100)))  # Rough approximation
            fractal_dim = np.log(n_boxes) / np.log(len(x))
            return float(fractal_dim)
        except:
            return 1.0  # Default to 1D

    def _estimate_lyapunov_exponent(self, func: Callable[[float], float],
                                   domain: Tuple[float, float]) -> float:
        """Estimate Lyapunov exponent for the function"""
        # Simplified Lyapunov exponent estimation
        # In practice, this requires more sophisticated analysis
        try:
            x0 = (domain[0] + domain[1]) / 2
            perturbation = 1e-6

            # Iterate function and track divergence
            x1, x2 = x0, x0 + perturbation
            divergence = 0

            for i in range(100):
                x1, x2 = func(x1), func(x2)
                if abs(x1) < 1e-10 or abs(x2) < 1e-10:
                    break
                current_divergence = abs(x2 - x1)
                if current_divergence > 0:
                    divergence += np.log(current_divergence)
                x2 = x1 + perturbation  # Reset perturbation

            lyapunov = divergence / 100 if divergence != 0 else 0
            return float(lyapunov)

        except Exception as e:
            self.logger.warning(f"Lyapunov exponent estimation failed: {e}")
            return 0.0

    def _compute_complexity_measures(self, data: np.ndarray) -> Dict[str, float]:
        """Compute various complexity measures"""
        try:
            # Approximate entropy (simplified)
            def approximate_entropy(data, m=2, r=0.2):
                N = len(data)
                if N < m + 1:
                    return 0

                # Normalize data
                data = (data - np.mean(data)) / np.std(data)

                # Compute correlation integrals
                def correlation_integral(m):
                    count = 0
                    for i in range(N - m + 1):
                        for j in range(N - m + 1):
                            if i != j:
                                dist = np.max(np.abs(data[i:i+m] - data[j:j+m]))
                                if dist < r:
                                    count += 1
                    return count / ((N - m + 1) * (N - m))

                c_m = correlation_integral(m)
                c_m_plus_1 = correlation_integral(m + 1)

                if c_m == 0 or c_m_plus_1 == 0:
                    return 0

                return -np.log(c_m_plus_1 / c_m)

            # Sample entropy
            samp_en = approximate_entropy(data)

            # Hurst exponent (simplified)
            # This is a very simplified calculation
            cumsum = np.cumsum(data - np.mean(data))
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(data)
            hurst = np.log(R / S) / np.log(len(data)) if S > 0 else 0.5

            return {
                'approximate_entropy': float(samp_en),
                'hurst_exponent': float(hurst),
                'complexity_index': float(samp_en * hurst)
            }

        except Exception as e:
            self.logger.warning(f"Complexity measures failed: {e}")
            return {'error': str(e)}

    def _analyze_chaos_indicators(self, func: Callable[[float], float],
                                domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze chaos indicators"""
        try:
            # Generate time series
            x0 = (domain[0] + domain[1]) / 2
            n_iterations = 1000

            trajectory = [x0]
            for _ in range(n_iterations):
                x_new = func(trajectory[-1])
                trajectory.append(x_new)

            # Compute chaos indicators
            chaos_indicators = {
                'lyapunov_exponent': self._estimate_lyapunov_exponent(func, domain),
                'bifurcation_analysis': self._analyze_bifurcation_simple(func, domain),
                'sensitivity_analysis': self._analyze_sensitivity(func, domain)
            }

            return chaos_indicators

        except Exception as e:
            self.logger.warning(f"Chaos analysis failed: {e}")
            return {'error': str(e)}

    def _analyze_bifurcation_simple(self, func: Callable[[float], float],
                                   domain: Tuple[float, float]) -> Dict[str, Any]:
        """Simple bifurcation analysis"""
        # This is a simplified bifurcation analysis
        # In practice, this would require parameter continuation methods
        return {'bifurcations_found': 0, 'method': 'simplified'}

    def _analyze_sensitivity(self, func: Callable[[float], float],
                           domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze sensitivity to initial conditions"""
        try:
            x0 = (domain[0] + domain[1]) / 2
            perturbation = 1e-6

            # Run trajectories
            x1, x2 = x0, x0 + perturbation

            divergence_history = []
            for _ in range(100):
                x1, x2 = func(x1), func(x2)
                divergence = abs(x2 - x1)
                divergence_history.append(divergence)
                x2 = x1 + perturbation  # Reset perturbation

            # Compute sensitivity measure
            avg_divergence = np.mean(divergence_history)
            max_divergence = np.max(divergence_history)

            return {
                'average_divergence': float(avg_divergence),
                'maximum_divergence': float(max_divergence),
                'sensitivity_measure': float(max_divergence / perturbation)
            }

        except Exception as e:
            return {'error': str(e)}

    def _compute_information_measures(self, data: np.ndarray) -> Dict[str, float]:
        """Compute information-theoretic measures"""
        try:
            # Shannon entropy (simplified)
            hist, _ = np.histogram(data, bins=50, density=True)
            hist = hist[hist > 0]  # Remove zeros
            shannon_entropy = -np.sum(hist * np.log2(hist))

            # Approximate Kolmogorov complexity (very simplified)
            # This is a rough approximation
            n = len(data)
            if n > 1:
                diffs = np.abs(np.diff(data))
                complexity = np.sum(diffs > np.std(diffs)) / n
            else:
                complexity = 0

            return {
                'shannon_entropy': float(shannon_entropy),
                'kolmogorov_complexity': float(complexity),
                'information_density': float(shannon_entropy / np.log2(n) if n > 1 else 0)
            }

        except Exception as e:
            return {'error': str(e)}

    def _create_analysis_visualizations(self, func: Callable[[float], float],
                                      domain: Tuple[float, float],
                                      results: Dict[str, Any]):
        """Create comprehensive analysis visualizations"""
        try:
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            fig.suptitle('Comprehensive Function Analysis', fontsize=16)

            x = np.linspace(domain[0], domain[1], 1000)
            y = func(x)

            # Plot 1: Function and derivatives
            axes[0, 0].plot(x, y, 'b-', linewidth=2, label='f(x)')
            axes[0, 0].set_title('Function')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()

            # Plot 2: Statistical distribution
            if 'statistical' in results:
                axes[0, 1].hist(y, bins=50, alpha=0.7, edgecolor='black', density=True)
                axes[0, 1].set_title('Distribution')
                axes[0, 1].set_xlabel('Value')
                axes[0, 1].set_ylabel('Density')

            # Plot 3: Autocorrelation
            if len(y) > 10:
                correlation = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
                correlation = correlation[correlation.size // 2:]
                lags = range(len(correlation))
                axes[0, 2].plot(lags, correlation)
                axes[0, 2].set_title('Autocorrelation')
                axes[0, 2].set_xlabel('Lag')
                axes[0, 2].set_ylabel('Correlation')
                axes[0, 2].grid(True, alpha=0.3)

            # Plot 4: Phase space (if applicable)
            axes[1, 0].plot(y[:-1], y[1:], 'r.', alpha=0.6, markersize=1)
            axes[1, 0].set_title('Phase Space')
            axes[1, 0].set_xlabel('x(n)')
            axes[1, 0].set_ylabel('x(n+1)')
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 5: Fourier analysis
            if len(y) > 10:
                fft_result = np.fft.fft(y)
                freq = np.fft.fftfreq(len(y))
                axes[1, 1].plot(freq[:len(freq)//2], np.abs(fft_result)[:len(freq)//2])
                axes[1, 1].set_title('Frequency Spectrum')
                axes[1, 1].set_xlabel('Frequency')
                axes[1, 1].set_ylabel('Power')
                axes[1, 1].grid(True, alpha=0.3)

            # Plot 6: Complexity measures
            if 'advanced' in results and 'complexity_measures' in results['advanced']:
                complexity = results['advanced']['complexity_measures']
                if 'error' not in complexity:
                    labels = list(complexity.keys())
                    values = list(complexity.values())
                    axes[1, 2].bar(labels, values, alpha=0.7)
                    axes[1, 2].set_title('Complexity Measures')
                    axes[1, 2].tick_params(axis='x', rotation=45)

            # Plot 7: Statistical properties
            if 'statistical' in results:
                stats = results['statistical']
                labels = ['Mean', 'Std', 'Skewness', 'Kurtosis']
                values = [stats.get('mean', 0), stats.get('std', 0),
                         stats.get('skewness', 0), stats.get('kurtosis', 0)]
                axes[2, 0].bar(labels, values, alpha=0.7, color='green')
                axes[2, 0].set_title('Statistical Properties')
                axes[2, 0].tick_params(axis='x', rotation=45)

            # Plot 8: Distribution fit
            if 'statistical' in results and 'distribution_fit' in results['statistical']:
                fits = results['statistical']['distribution_fit']
                if 'best_fit' in fits:
                    best_dist = fits['best_fit']
                    axes[2, 1].text(0.5, 0.5, f'Best Fit:\\n{best_dist}',
                                   transform=axes[2, 1].transAxes,
                                   fontsize=12, ha='center', va='center')
                    axes[2, 1].set_title('Distribution Analysis')

            # Plot 9: Summary
            axes[2, 2].axis('off')
            summary_text = f"""
Analysis Summary:

Domain: [{domain[0]:.2f}, {domain[1]:.2f}]
Range: [{np.min(y):.2f}, {np.max(y):.2f}]
Zeros: {len(results.get('basic', {}).get('zeros', []))}
Monotonic: {results.get('basic', {}).get('monotonicity', {}).get('type', 'unknown')}
Bounded: {results.get('basic', {}).get('boundedness', {}).get('bounded', 'unknown')}
"""

            axes[2, 2].text(0.1, 0.9, summary_text, transform=axes[2, 2].transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

            plt.tight_layout()
            plt.savefig(self.output_dir / 'comprehensive_analysis.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

            self.logger.info(f"Comprehensive analysis visualization saved to {self.output_dir}")

        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")

    def _generate_comprehensive_report(self, func: Callable[[float], float],
                                     domain: Tuple[float, float],
                                     results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
# Comprehensive Mathematical Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Function Analysis Summary

### Basic Properties
- **Domain**: [{domain[0]:.2f}, {domain[1]:.2f}]
"""

        if 'basic' in results:
            basic = results['basic']
            report += f"- **Range**: [{basic['range'][0]:.2f}, {basic['range'][1]:.2f}]\n"
            report += f"- **Zeros**: {len(basic['zeros'])} found\n"
            report += f"- **Monotonicity**: {basic['monotonicity']['type']} (confidence: {basic['monotonicity']['confidence']:.2f})\n"
            report += f"- **Boundedness**: {basic['boundedness']['bounded']}\n"

            if basic['periodicity']:
                report += f"- **Period**: {basic['periodicity']:.4f}\n"

            if basic['symmetry']['type'] != 'neither':
                report += f"- **Symmetry**: {basic['symmetry']['type']} (confidence: {basic['symmetry']['confidence']:.2f})\n"

        if 'calculus' in results:
            calculus = results['calculus']
            if 'error' not in calculus:
                report += f"""
### Calculus Properties
- **Derivative at center**: {calculus.get('derivative_at_center', 'N/A'):.4f}
- **Integral over domain**: {calculus.get('integral_over_domain', 'N/A'):.4f}
- **Number of extrema**: {len(calculus.get('local_extrema', []))}
"""

        if 'statistical' in results:
            stats = results['statistical']
            report += f"""
### Statistical Properties
- **Mean**: {stats.get('mean', 'N/A'):.4f}
- **Standard Deviation**: {stats.get('std', 'N/A'):.4f}
- **Skewness**: {stats.get('skewness', 'N/A'):.4f}
- **Kurtosis**: {stats.get('kurtosis', 'N/A'):.4f}
"""

            if 'distribution_fit' in stats and 'best_fit' in stats['distribution_fit']:
                report += f"- **Best distribution fit**: {stats['distribution_fit']['best_fit']}\n"

        if 'advanced' in results:
            advanced = results['advanced']
            if 'complexity_measures' in advanced and 'error' not in advanced['complexity_measures']:
                complexity = advanced['complexity_measures']
                report += f"""
### Advanced Properties
- **Approximate Entropy**: {complexity.get('approximate_entropy', 'N/A'):.4f}
- **Hurst Exponent**: {complexity.get('hurst_exponent', 'N/A'):.4f}
- **Lyapunov Exponent**: {advanced.get('lyapunov_exponent', 'N/A'):.4f}
"""

        report += f"""
## Analysis History
Total operations: {len(self.analysis_history)}
Successful operations: {sum(1 for op in self.analysis_history if op['success'])}
Failed operations: {sum(1 for op in self.analysis_history if not op['success'])}
"""

        return report

    def save_analysis_results(self, results: Dict[str, Any],
                            filename: str = "analysis_results") -> Path:
        """Save analysis results to files"""
        # Save JSON results
        json_path = self.output_dir / f"{filename}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save text report
        if 'report' in results:
            report_path = self.output_dir / f"{filename}_report.txt"
            with open(report_path, 'w') as f:
                f.write(results['report'])

        self.logger.info(f"Analysis results saved to {self.output_dir}")
        return self.output_dir

    def create_analysis_gallery(self) -> Dict[str, Any]:
        """Create a gallery of different analysis examples"""
        console.print(Panel.fit("🔬 Comprehensive Mathematical Analysis Gallery", style="bold magenta"))

        gallery_results = {}

        # Example 1: Trigonometric function
        console.print("\\n1. 📊 Trigonometric Function Analysis", style="bold blue")
        def trig_func(x):
            return np.sin(2 * np.pi * x) * np.exp(-x/5)

        trig_results = self.comprehensive_function_analysis(
            trig_func, (0, 10), "Trigonometric Function"
        )
        gallery_results['trigonometric'] = trig_results
        console.print("   ✅ Completed trigonometric function analysis")

        # Example 2: Polynomial function
        console.print("\\n2. 📈 Polynomial Function Analysis", style="bold blue")
        def poly_func(x):
            return x**3 - 2*x**2 + x - 1

        poly_results = self.comprehensive_function_analysis(
            poly_func, (-2, 3), "Cubic Polynomial"
        )
        gallery_results['polynomial'] = poly_results
        console.print("   ✅ Completed polynomial function analysis")

        # Example 3: Exponential function
        console.print("\\n3. ⚡ Exponential Function Analysis", style="bold blue")
        def exp_func(x):
            return np.exp(-x) * np.sin(5*x)

        exp_results = self.comprehensive_function_analysis(
            exp_func, (0, 5), "Damped Oscillation"
        )
        gallery_results['exponential'] = exp_results
        console.print("   ✅ Completed exponential function analysis")

        # Save gallery results
        self.save_analysis_results(gallery_results, "analysis_gallery")

        console.print("\\n🎉 Comprehensive Analysis Gallery Complete!")
        console.print(f"   📁 All results saved to: {self.output_dir}")
        console.print(f"   📊 Total functions analyzed: {len(gallery_results)}")

        return gallery_results


def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Mathematical Analysis")
    parser.add_argument('--function', help='Function to analyze (Python expression)')
    parser.add_argument('--domain', default='-5,5', help='Domain as "min,max"')
    parser.add_argument('--output', help='Output directory')
    parser.add_argument('--analysis-type', choices=['basic', 'calculus', 'statistical', 'advanced', 'full'],
                       default='full', help='Type of analysis')
    parser.add_argument('--gallery', action='store_true', help='Create analysis gallery')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # Setup analyzer
    output_dir = args.output if args.output else "comprehensive_analysis"
    analyzer = ComprehensiveMathematicalAnalyzer(
        output_dir=output_dir,
        log_level="DEBUG" if args.verbose else "INFO"
    )

    if args.gallery:
        # Create analysis gallery
        results = analyzer.create_analysis_gallery()
        return 0

    elif args.function:
        # Analyze specific function
        def func(x):
            return eval(args.function, {"__builtins__": {}, "np": np, "x": x})

        domain = tuple(map(float, args.domain.split(',')))

        results = analyzer.comprehensive_function_analysis(
            func, domain, args.analysis_type
        )

        # Save results
        analyzer.save_analysis_results(results)

        print(f"Analysis completed! Results saved to: {analyzer.output_dir}")
        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    exit(main())
