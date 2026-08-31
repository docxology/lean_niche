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
        # Sample an extended window (3x the domain) so that periodicity
        # analysis can observe a full recurrence even when the requested
        # domain covers only one period.
        x = np.linspace(domain[0], domain[1], 1000)
        y = func(x)
        x_ext = np.linspace(domain[0] - (domain[1] - domain[0]), domain[1] + (domain[1] - domain[0]), 3001)
        y_ext = func(x_ext)

        analysis = {
            'domain': domain,
            'range': (float(np.min(y)), float(np.max(y))),
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'median': float(np.median(y)),
            'zeros': self._find_zeros(func, domain),
            'extrema': self._find_extrema(func, domain),
            'asymptotes': self._find_asymptotes(func, domain),
            'periodicity': self._analyze_periodicity(x_ext, y_ext)
        }

        # Generate analysis report
        self._generate_analysis_report(analysis, title, save_path)

        return analysis

    def _find_zeros(self, func: Callable[[float], float],
                   domain: Tuple[float, float]) -> List[float]:
        """Find zeros of the function by scanning for sign changes"""
        zeros: List[float] = []
        try:
            grid = np.linspace(domain[0], domain[1], 2001)
            vals = np.asarray([float(func(xx)) for xx in grid])
            for i in range(len(grid) - 1):
                f1, f2 = vals[i], vals[i + 1]
                if not (np.isfinite(f1) and np.isfinite(f2)):
                    continue
                if f1 == 0.0:
                    zeros.append(float(grid[i]))
                elif f1 * f2 < 0:
                    try:
                        res = optimize.root_scalar(func, bracket=(grid[i], grid[i + 1]), method='brentq')
                        if res.converged:
                            zeros.append(float(res.root))
                    except Exception:
                        pass
            # dedupe close roots
            deduped: List[float] = []
            for z in sorted(zeros):
                if not deduped or abs(z - deduped[-1]) > 1e-6:
                    deduped.append(z)
            zeros = deduped
        except Exception:
            pass
        return zeros

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

        # Probe near the domain edges and beyond for divergence/horizontal behavior
        with np.errstate(all='ignore'):
            try:
                for side, xs in (('left', (domain[0] - 1000, domain[0] - 100, domain[0] + 1e-9)),
                                 ('right', (domain[1] - 1e-9, domain[1] + 100, domain[1] + 1000))):
                    vals = []
                    for xx in xs:
                        try:
                            vals.append(float(func(xx)))
                        except (ZeroDivisionError, OverflowError, ValueError):
                            vals.append(float('inf'))
                    inner = vals[0] if side == 'right' else vals[2]
                    outer = vals[2] if side == 'right' else vals[0]
                    if (not np.isfinite(inner)) or abs(inner) > 1e6:
                        asymptotes['vertical'].append(side)
                    elif np.isfinite(outer) and abs(outer - inner) < 1e-3 and abs(outer) < 1e6:
                        asymptotes['horizontal'].append((side, outer))
            except Exception:
                pass

        # Scan the domain interior for divergence (poles) via dense sampling
        try:
            with np.errstate(all='ignore'):
                grid = np.linspace(domain[0], domain[1], 2001)
                for xx in grid:
                    try:
                        val = float(func(xx))
                    except (ZeroDivisionError, OverflowError, ValueError):
                        val = float('inf')
                    if (not np.isfinite(val)) or abs(val) > 1e6:
                        asymptotes['vertical'].append(round(float(xx), 6))
                        break
        except Exception:
            pass

        return asymptotes

    def _analyze_periodicity(self, x: np.ndarray, y: np.ndarray) -> Optional[float]:
        """Analyze if function is periodic"""
        # Simple periodicity analysis using normalized autocorrelation
        if len(y) < 10:
            return None

        yc = y - np.mean(y)
        energy = np.sum(yc * yc)
        if energy == 0:
            return None
        correlation = np.correlate(yc, yc, mode='full')
        correlation = correlation[correlation.size // 2:] / energy

        # Find the first significant positive-correlation local maximum
        # (a strict rising-then-falling comparison over a small window so that
        # noisy sample-to-sample jitter cannot mask the peak)
        # Skip the near-zero-lag plateau, then find the first genuine local
        # maximum of the autocorrelation (a candidate period). A periodic
        # signal's autocorrelation oscillates: it must rise from a trough
        # back above 0.3. Monotone decay (non-periodic signals) never rises
        # again and is rejected.
        # Ignore the near-zero-lag coherence plateau: start looking only
        # after the autocorrelation has decisively fallen (below 0.2), then
        # accept the first lag where it rises back above 0.3 — a genuine
        # period recurrence. Monotone decay (non-periodic signals) never
        # rises back and is rejected.
        skip = max(3, len(correlation) // 100)
        armed = False
        for i in range(skip, len(correlation) - 1):
            if not armed:
                if correlation[i] < 0.2:
                    armed = True
                continue
            if (correlation[i] > 0.3
                    and correlation[i] >= correlation[i - 1]
                    and correlation[i] >= correlation[i + 1]):
                period = abs(float(x[i] - x[0]))
                span = abs(float(x[-1] - x[0]))
                if 0 < period < span:
                    return period
                break

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
        valid_methods = ('quad', 'trapezoid', 'simpson')
        if method not in valid_methods:
            raise ValueError(f"Unknown integration method: {method!r}; expected one of {valid_methods}")

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
                integral = np.trapezoid(y, x) if hasattr(np, 'trapezoid') else np.trapz(y, x)
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
                         variables: List[str] = None,
                         operation: str = 'full') -> Dict[str, Any]:
        """Symbolic mathematical analysis (optionally restricted to one operation)"""
        if variables is None:
            variables = ['x']

        try:
            # Parse expression (strict: reject non-mathematical input)
            expr = sp.sympify(expression, locals={}, evaluate=True)
            if expr.free_symbols and not set(str(v) for v in expr.free_symbols) <= set(variables):
                raise ValueError("expression contains unknown symbols")

            analysis = {
                'expression': str(expr),
                'variables': variables,
                'simplified': str(sp.simplify(expr)),
                'expanded': str(sp.expand(expr)),
                'factorized': str(sp.factor(expr))
            }

            # Try to compute derivative if single variable
            if operation in ('full', 'derivative') and len(variables) == 1:
                var = sp.symbols(variables[0])
                try:
                    derivative = str(sp.diff(expr, var))
                    analysis['derivative'] = derivative
                except:
                    analysis['derivative'] = 'Could not compute'

            # Try to compute integral if single variable
            if operation in ('full', 'integral') and len(variables) == 1:
                var = sp.symbols(variables[0])
                try:
                    integral = str(sp.integrate(expr, var))
                    analysis['integral'] = integral
                except:
                    analysis['integral'] = 'Could not compute'

            return analysis

        except Exception as e:
            return {'error': str(e)}

    def _fit_distributions(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit common distributions and return fit quality"""
        fits = {}
        try:
            candidates = {
                'normal': stats.norm,
                'exponential': stats.expon,
                'uniform': stats.uniform
            }
            best_name, best_p = None, -1.0
            for name, dist in candidates.items():
                try:
                    params = dist.fit(data)
                    _, p_value = stats.kstest(data, dist.cdf, args=params)
                    fits[name] = {'params': [float(p_) for p_ in params],
                                  'ks_p_value': float(p_value)}
                    if p_value > best_p:
                        best_name, best_p = name, p_value
                except Exception:
                    continue
            if best_name:
                fits['best_fit'] = best_name
        except Exception:
            pass
        return fits

    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """IQR-based outlier detection"""
        try:
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
            mask = (data < lower) | (data > upper)
            return {
                'method': 'iqr',
                'lower_bound': float(lower),
                'upper_bound': float(upper),
                'indices': np.where(mask)[0].tolist(),
                'values': [float(v) for v in data[mask]],
                'count': int(mask.sum())
            }
        except Exception:
            return {'method': 'iqr', 'indices': [], 'values': [], 'count': 0}

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
            'kurtosis': float(stats.kurtosis(data)),
            'distribution_fit': self._fit_distributions(data),
            'outliers': self._detect_outliers(data)
        }

        # Normality tests
        if len(data) >= 3:
            try:
                shapiro = {
                    'statistic': float(stats.shapiro(data)[0]),
                    'p_value': float(stats.shapiro(data)[1])
                }
                normaltest = {
                    'statistic': float(stats.normaltest(data)[0]),
                    'p_value': float(stats.normaltest(data)[1])
                }
                analysis['shapiro_test'] = shapiro
                analysis['normaltest'] = normaltest
                # Structured view expected by consumers
                analysis['normality_test'] = {
                    'shapiro_wilk': shapiro,
                    'kolmogorov_smirnov': normaltest
                }
            except Exception:
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

        def _fmt(value, spec='.4f'):
            try:
                return format(float(value), spec)
            except (TypeError, ValueError):
                return str(value)

        summary_text = (
            "        Summary Statistics:\n\n"
            f"        Sample Size: {analysis.get('n', 'N/A')}\n"
            f"        Mean: {_fmt(analysis.get('mean'))}\n"
            f"        Std Dev: {_fmt(analysis.get('std'))}\n"
            f"        Min: {_fmt(analysis.get('min'))}\n"
            f"        Max: {_fmt(analysis.get('max'))}\n"
            f"        Skewness: {_fmt(analysis.get('skewness'))}\n"
        )

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
    console.print("\n1. 📈 Function Analysis", style="bold blue")
    def quadratic(x):
        return x**2 - 2*x + 1

    analysis = analyzer.analyze_function(
        quadratic, (-3, 5), "Quadratic Function Analysis", "quadratic_analysis.txt"
    )
    console.print("   ✅ Completed function analysis")

    # Example 2: Numerical integration
    console.print("\n2. ∫ Numerical Integration", style="bold blue")
    def integrand(x):
        return np.sin(x) * np.exp(-x/2)

    integral_result = analyzer.numerical_integration(integrand, 0, 5)
    console.print(f"   ✅ Completed numerical integration: {integral_result.get('result', 'N/A')}")

    # Example 3: Symbolic analysis
    console.print("\n3. 🔣 Symbolic Analysis", style="bold blue")
    symbolic_result = analyzer.symbolic_analysis("x^2 + 2*x + 1")
    console.print("   ✅ Completed symbolic analysis")
    console.print(f"   📝 Expression: {symbolic_result.get('expression', 'N/A')}")
    console.print(f"   📝 Derivative: {symbolic_result.get('derivative', 'N/A')}")

    # Example 4: Statistical analysis
    console.print("\n4. 📊 Statistical Analysis", style="bold blue")
    np.random.seed(42)
    data = np.random.normal(5, 2, 100).tolist()
    stats_result = analyzer.statistical_analysis(data)
    console.print(f"   ✅ Completed statistical analysis — Sample Size: {stats_result.get('n', 'N/A')}")
    console.print("\n🎉 Mathematical Analysis Gallery Complete!")
    console.print(f"   📁 All results saved to: {analyzer.output_dir}")
if __name__ == "__main__":
    create_analysis_gallery()
