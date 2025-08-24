"""Mathematical Analysis Tools

Advanced mathematical analysis and computation tools for LeanNiche.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Callable, Optional
from pathlib import Path
import pandas as pd
from scipy import optimize, integrate, special, stats
import sympy as sp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class MathematicalAnalyzer:
    """Advanced mathematical analysis tools"""

    def __init__(self):
        """Initialize the analyzer"""
        self.output_dir = Path("analysis_results")
        self.output_dir.mkdir(exist_ok=True)

    def analyze_function(self, func: Callable[[float], float],
                        domain: Tuple[float, float],
                        title: str = "Function Analysis",
                        save_path: Optional[str] = None) -> Dict[str, Any]:
        """Comprehensive function analysis"""
        x = np.linspace(domain[0], domain[1], 1000)
        y = func(x)

        analysis = {
            'domain': domain,
            'range': (float(np.min(y)), float(np.max(y))),
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'median': float(np.median(y)),
            'zeros': self._find_zeros(func, domain),
            'extrema': self._find_extrema(func, domain),
            'asymptotes': self._find_asymptotes(func, domain),
            'periodicity': self._analyze_periodicity(x, y)
        }

        # Generate analysis report
        self._generate_analysis_report(analysis, title, save_path)

        return analysis

    def _find_zeros(self, func: Callable[[float], float],
                   domain: Tuple[float, float]) -> List[float]:
        """Find zeros of the function"""
        try:
            # Use scipy's root finding
            result = optimize.root_scalar(func, bracket=domain, method='brentq')
            if result.converged:
                return [float(result.root)]
        except:
            pass
        return []

    def _find_extrema(self, func: Callable[[float], float],
                     domain: Tuple[float, float]) -> Dict[str, List[Tuple[float, float]]]:
        """Find local extrema"""
        extrema = {'maxima': [], 'minima': []}

        try:
            # Find critical points by minimizing the derivative
            def negative_func(x):
                return -func(x)

            # Find minimum
            min_result = optimize.minimize_scalar(func, bounds=domain, method='bounded')
            if min_result.success:
                extrema['minima'].append((float(min_result.x), float(min_result.fun)))

            # Find maximum
            max_result = optimize.minimize_scalar(negative_func, bounds=domain, method='bounded')
            if max_result.success:
                extrema['maxima'].append((float(max_result.x), float(-max_result.fun)))

        except:
            pass

        return extrema

    def _find_asymptotes(self, func: Callable[[float], float],
                        domain: Tuple[float, float]) -> Dict[str, Any]:
        """Analyze asymptotic behavior"""
        asymptotes = {
            'horizontal': [],
            'vertical': [],
            'oblique': []
        }

        # Check for horizontal asymptotes
        try:
            limit_left = func(domain[0])
            limit_right = func(domain[1])
            asymptotes['horizontal'] = [
                ('left', float(limit_left)),
                ('right', float(limit_right))
            ]
        except:
            pass

        return asymptotes

    def _analyze_periodicity(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Analyze if function is periodic"""
        # Simple periodicity analysis using autocorrelation
        if len(y) < 10:
            return None

        # Compute autocorrelation
        correlation = np.correlate(y - np.mean(y), y - np.mean(y), mode='full')
        correlation = correlation[correlation.size // 2:]

        # Find peaks in autocorrelation
        peaks = []
        for i in range(1, len(correlation) - 1):
            if correlation[i] > correlation[i-1] and correlation[i] > correlation[i+1]:
                if correlation[i] > 0.5 * correlation[0]:  # Significant peak
                    peaks.append(i)

        if peaks:
            # Estimate period as first significant peak
            period_idx = peaks[0]
            period = abs(x[period_idx] - x[0])
            return float(period)

        return None

    def _generate_analysis_report(self, analysis: Dict[str, Any],
                                title: str, save_path: Optional[str] = None):
        """Generate analysis report"""
        table = Table(title=f"📊 {title}")
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Domain", ".2f")
        table.add_row("Range", ".2f")
        table.add_row("Mean", ".4f")
        table.add_row("Standard Deviation", ".4f")
        table.add_row("Median", ".4f")

        if analysis['zeros']:
            table.add_row("Zeros", ",.2f")
        else:
            table.add_row("Zeros", "None found")

        if analysis['extrema']['maxima']:
            table.add_row("Local Maxima", ",.2f")
        if analysis['extrema']['minima']:
            table.add_row("Local Minima", ",.2f")

        if analysis['periodicity']:
            table.add_row("Period", ".4f")
        else:
            table.add_row("Period", "Not periodic")

        console.print(table)

        if save_path:
            with open(self.output_dir / save_path, 'w') as f:
                f.write(f"# {title}\\n\\n")
                f.write(f"Domain: {analysis['domain']}\\n")
                f.write(f"Range: {analysis['range']}\\n")
                f.write(f"Mean: {analysis['mean']:.4f}\\n")
                f.write(f"Std: {analysis['std']:.4f}\\n")
                f.write(f"Median: {analysis['median']:.4f}\\n")
                f.write(f"Zeros: {analysis['zeros']}\\n")
                f.write(f"Extrema: {analysis['extrema']}\\n")
                f.write(f"Periodicity: {analysis['periodicity']}\\n")

    def numerical_integration(self, func: Callable[[float], float],
                            a: float, b: float,
                            method: str = 'quad') -> Dict[str, Any]:
        """Numerical integration using various methods"""
        result = {}

        try:
            if method == 'quad':
                integral, error = integrate.quad(func, a, b)
                result = {
                    'method': 'quad',
                    'result': float(integral),
                    'error': float(error),
                    'absolute_error': abs(error)
                }
            elif method == 'trapezoid':
                x = np.linspace(a, b, 1000)
                y = func(x)
                integral = np.trapz(y, x)
                result = {
                    'method': 'trapezoid',
                    'result': float(integral),
                    'points': len(x)
                }
            elif method == 'simpson':
                x = np.linspace(a, b, 1000)
                y = func(x)
                integral = integrate.simpson(y, x=x)
                result = {
                    'method': 'simpson',
                    'result': float(integral),
                    'points': len(x)
                }
        except Exception as e:
            result = {'error': str(e)}

        return result

    def symbolic_analysis(self, expression: str,
                         variables: List[str] = None) -> Dict[str, Any]:
        """Symbolic mathematical analysis"""
        if variables is None:
            variables = ['x']

        try:
            # Parse expression
            expr = sp.sympify(expression)

            analysis = {
                'expression': str(expr),
                'variables': variables,
                'simplified': str(sp.simplify(expr)),
                'expanded': str(sp.expand(expr)),
                'factorized': str(sp.factor(expr))
            }

            # Try to compute derivative if single variable
            if len(variables) == 1:
                var = sp.symbols(variables[0])
                try:
                    derivative = str(sp.diff(expr, var))
                    analysis['derivative'] = derivative
                except:
                    analysis['derivative'] = 'Could not compute'

            # Try to compute integral if single variable
            if len(variables) == 1:
                var = sp.symbols(variables[0])
                try:
                    integral = str(sp.integrate(expr, var))
                    analysis['integral'] = integral
                except:
                    analysis['integral'] = 'Could not compute'

            return analysis

        except Exception as e:
            return {'error': str(e)}

    def statistical_analysis(self, data: List[float],
                           alpha: float = 0.05) -> Dict[str, Any]:
        """Comprehensive statistical analysis"""
        data = np.array(data)

        analysis = {
            'n': len(data),
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'mode': float(stats.mode(data)[0]),
            'std': float(np.std(data)),
            'var': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'range': float(np.max(data) - np.min(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'iqr': float(np.percentile(data, 75) - np.percentile(data, 25)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }

        # Normality tests
        if len(data) >= 3:
            try:
                analysis['shapiro_test'] = {
                    'statistic': float(stats.shapiro(data)[0]),
                    'p_value': float(stats.shapiro(data)[1])
                }
                analysis['normaltest'] = {
                    'statistic': float(stats.normaltest(data)[0]),
                    'p_value': float(stats.normaltest(data)[1])
                }
            except:
                pass

        # Confidence intervals
        if len(data) > 1:
            try:
                confidence_interval = stats.t.interval(alpha, len(data)-1,
                                                    loc=np.mean(data),
                                                    scale=stats.sem(data))
                analysis['confidence_interval'] = {
                    'alpha': alpha,
                    'lower': float(confidence_interval[0]),
                    'upper': float(confidence_interval[1])
                }
            except:
                pass

        return analysis

    def plot_analysis_results(self, analysis: Dict[str, Any],
                            title: str = "Analysis Results",
                            save_path: Optional[str] = None):
        """Create visualization of analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)

        # Plot 1: Statistical distribution
        if 'data' in analysis:
            data = analysis['data']
            axes[0, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title("Distribution")
            axes[0, 0].set_xlabel("Value")
            axes[0, 0].set_ylabel("Frequency")

        # Plot 2: Box plot
        if 'data' in analysis:
            data = analysis['data']
            axes[0, 1].boxplot(data)
            axes[0, 1].set_title("Box Plot")
            axes[0, 1].set_ylabel("Value")

        # Plot 3: Q-Q plot
        if 'data' in analysis:
            data = analysis['data']
            stats.probplot(data, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title("Q-Q Plot")

        # Plot 4: Summary statistics
        axes[1, 1].axis('off')
        summary_text = f"""
        Summary Statistics:

        Sample Size: {analysis.get('n', 'N/A')}
        Mean: {analysis.get('mean', 'N/A'):.4f}
        Std Dev: {analysis.get('std', 'N/A'):.4f}
        Min: {analysis.get('min', 'N/A'):.4f}
        Max: {analysis.get('max', 'N/A'):.4f}
        Skewness: {analysis.get('skewness', 'N/A'):.4f}
        """

        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig


def create_analysis_gallery():
    """Create a gallery of mathematical analysis examples"""
    console.print(Panel.fit("🔬 LeanNiche Mathematical Analysis Gallery", style="bold magenta"))

    analyzer = MathematicalAnalyzer()

    # Example 1: Function analysis
    console.print("\\n1. 📈 Function Analysis", style="bold blue")
    def quadratic(x):
        return x**2 - 2*x + 1

    analysis = analyzer.analyze_function(quadratic, (-3, 5), "Quadratic Function Analysis", "quadratic_analysis.txt")
    console.print("   ✅ Completed function analysis"    console.print(".2f"    console.print(".4f"
    # Example 2: Numerical integration
    console.print("\\n2. ∫ Numerical Integration", style="bold blue")
    def integrand(x):
        return np.sin(x) * np.exp(-x/2)

    integral_result = analyzer.numerical_integration(integrand, 0, 5)
    console.print("   ✅ Completed numerical integration"    console.print(".4f"
    # Example 3: Symbolic analysis
    console.print("\\n3. 🔣 Symbolic Analysis", style="bold blue")
    symbolic_result = analyzer.symbolic_analysis("x^2 + 2*x + 1")
    console.print("   ✅ Completed symbolic analysis"    console.print(f"   📝 Expression: {symbolic_result.get('expression', 'N/A')}")
    console.print(f"   📝 Derivative: {symbolic_result.get('derivative', 'N/A')}")

    # Example 4: Statistical analysis
    console.print("\\n4. 📊 Statistical Analysis", style="bold blue")
    np.random.seed(42)
    data = np.random.normal(5, 2, 100).tolist()
    stats_result = analyzer.statistical_analysis(data)
    console.print("   ✅ Completed statistical analysis"    console.print(f"   📊 Sample Size: {stats_result['n']}")
    console.print(".4f"    console.print(".4f"
    console.print("\\n🎉 Mathematical Analysis Gallery Complete!"    console.print(f"   📁 All results saved to: {analyzer.output_dir}")


if __name__ == "__main__":
    create_analysis_gallery()
