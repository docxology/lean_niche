#!/usr/bin/env python3
"""
Statistical Analysis Module for LeanNiche

This module provides comprehensive statistical analysis including:
- Data generation for various distributions
- Descriptive statistics and hypothesis testing
- Confidence intervals and correlation analysis
- Statistical modeling and inference
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from scipy import stats
from dataclasses import dataclass


@dataclass
class StatisticalSummary:
    """Container for statistical summary results."""
    mean: float
    median: float
    std: float
    variance: float
    min_val: float
    max_val: float
    q25: float
    q75: float
    sample_size: int
    confidence_interval: Optional[Tuple[float, float]] = None


class DataGenerator:
    """Statistical data generation tools."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize data generator with optional random seed."""
        if seed is not None:
            np.random.seed(seed)

    def generate_sample_data(self, datasets_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate realistic sample data for analysis.

        Args:
            datasets_config: Configuration for each dataset to generate

        Returns:
            Dictionary containing generated datasets
        """
        print("📊 Generating sample datasets...")

        datasets = {}

        for name, config in datasets_config.items():
            if name == 'heights':
                # Human heights (normal distribution)
                heights = np.random.normal(170, 10, 100)
                datasets['heights'] = {
                    'data': heights,
                    'name': 'Human Heights (cm)',
                    'description': 'Normally distributed heights of 100 adults'
                }

            elif name == 'scores':
                # Exam scores (beta distribution)
                scores = np.random.beta(2, 5, 80) * 100
                datasets['scores'] = {
                    'data': scores,
                    'name': 'Exam Scores (%)',
                    'description': 'Beta-distributed exam scores of 80 students'
                }

            elif name == 'reaction_times':
                # Reaction times (exponential distribution)
                reaction_times = np.random.exponential(0.5, 60)
                datasets['reaction_times'] = {
                    'data': reaction_times,
                    'name': 'Reaction Times (s)',
                    'description': 'Exponentially distributed reaction times'
                }

            elif name == 'treatment':
                # Before/after treatment (paired data)
                before_treatment = np.random.normal(100, 15, 50)
                after_treatment = before_treatment + np.random.normal(-5, 8, 50)
                datasets['treatment'] = {
                    'data': {'before': before_treatment, 'after': after_treatment},
                    'name': 'Treatment Effect',
                    'description': 'Paired data showing treatment effect'
                }

        print(f"✅ Generated {len(datasets)} datasets")
        return datasets

    def generate_distribution_data(self, distribution: str, parameters: Dict[str, Any],
                                 sample_size: int = 1000) -> Dict[str, Any]:
        """Generate data from specific distributions."""
        if distribution == 'normal':
            data = np.random.normal(parameters.get('mean', 0), parameters.get('std', 1), sample_size)
        elif distribution == 'exponential':
            data = np.random.exponential(parameters.get('scale', 1), sample_size)
        elif distribution == 'beta':
            data = np.random.beta(parameters.get('a', 2), parameters.get('b', 5), sample_size)
        elif distribution == 'gamma':
            data = np.random.gamma(parameters.get('shape', 2), parameters.get('scale', 1), sample_size)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return {
            'data': data,
            'distribution': distribution,
            'parameters': parameters,
            'sample_size': sample_size
        }

    def generate_time_series_data(self, trend_type: str = 'linear',
                                seasonality: bool = True,
                                noise_std: float = 0.1,
                                length: int = 365) -> Dict[str, Any]:
        """Generate time series data with trend and seasonality."""
        t = np.arange(length)

        # Base trend
        if trend_type == 'linear':
            trend = 0.01 * t
        elif trend_type == 'exponential':
            trend = np.exp(0.001 * t) - 1
        else:
            trend = np.zeros(length)

        # Seasonality
        if seasonality:
            seasonal = np.sin(2 * np.pi * t / 52) + 0.5 * np.sin(2 * np.pi * t / 12)
        else:
            seasonal = np.zeros(length)

        # Noise
        noise = np.random.normal(0, noise_std, length)

        # Combine components
        data = trend + seasonal + noise

        return {
            'data': data,
            'time': t,
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise,
            'trend_type': trend_type,
            'seasonality': seasonality,
            'noise_std': noise_std
        }


class StatisticalAnalyzer:
    """Core statistical analysis tools."""

    def __init__(self):
        """Initialize statistical analyzer."""
        pass

    def perform_comprehensive_analysis(self, datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis on all datasets."""
        print("🔬 Performing comprehensive statistical analysis...")

        analysis_results = {}

        for name, dataset in datasets.items():
            print(f"  📈 Analyzing {dataset['name']}...")

            if name == 'treatment':
                # Paired t-test analysis
                before = dataset['data']['before']
                after = dataset['data']['after']

                analysis_results[name] = {
                    'dataset_info': dataset,
                    'analysis_type': 'paired_t_test',
                    'sample_size': len(before),
                    'before_stats': self._compute_basic_stats(before),
                    'after_stats': self._compute_basic_stats(after),
                    'difference': (after - before).tolist(),
                    'effect_size': float(np.mean(after - before) / np.std(after - before)),
                    'confidence_interval': self._compute_confidence_interval(after - before)
                }
            else:
                # Standard statistical analysis
                data = dataset['data']
                analysis_results[name] = {
                    'dataset_info': dataset,
                    'analysis_type': 'descriptive_statistics',
                    'sample_size': len(data),
                    'basic_stats': self._compute_basic_stats(data),
                    'distribution_test': self._test_distribution_normality(data),
                    'confidence_interval': self._compute_confidence_interval(data)
                }

        print(f"✅ Analyzed {len(datasets)} datasets")
        return analysis_results

    def _compute_basic_stats(self, data: Union[List, np.ndarray]) -> Dict[str, Any]:
        """Compute basic statistical measures."""
        data = np.array(data)
        return {
            'mean': float(np.mean(data)),
            'median': float(np.median(data)),
            'std': float(np.std(data)),
            'var': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75))
        }

    def _compute_confidence_interval(self, data: Union[List, np.ndarray],
                                   confidence: float = 0.95) -> Dict[str, Any]:
        """Compute confidence interval using t-distribution."""
        data = np.array(data)
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)

        # t-score for 95% confidence with n-1 degrees of freedom
        if n > 1:
            t_score = 1.96  # Approximation for large n
        else:
            t_score = 0

        margin = t_score * se
        return {
            'lower': float(mean - margin),
            'upper': float(mean + margin),
            'margin': float(margin),
            'confidence_level': confidence
        }

    def _test_distribution_normality(self, data: Union[List, np.ndarray]) -> Dict[str, Any]:
        """Test if data follows normal distribution."""
        data = np.array(data)

        if len(data) < 3:
            return {'test': 'shapiro', 'p_value': 1.0, 'normal': True}

        # Shapiro-Wilk test
        try:
            _, p_value = stats.shapiro(data)
            return {
                'test': 'shapiro',
                'p_value': float(p_value),
                'normal': p_value > 0.05,
                'skewness': float(stats.skew(data)),
                'kurtosis': float(stats.kurtosis(data))
            }
        except Exception:
            return {'test': 'shapiro', 'p_value': None, 'normal': None}

    def compute_correlations(self, data_list: List[Union[List, np.ndarray]]) -> List[List[float]]:
        """Compute correlation matrix between datasets."""
        # Ensure all arrays have the same length
        min_length = min(len(data) for data in data_list)
        trimmed_data = [np.array(data[:min_length]) for data in data_list]

        # Stack arrays properly
        combined_data = np.column_stack(trimmed_data)
        corr_matrix = np.corrcoef(combined_data.T)
        return corr_matrix.tolist()

    def hypothesis_test(self, data1: Union[List, np.ndarray], data2: Union[List, np.ndarray],
                      test_type: str = 't_test') -> Dict[str, Any]:
        """Perform hypothesis testing between two datasets."""
        data1 = np.array(data1)
        data2 = np.array(data2)

        if test_type == 't_test':
            t_stat, p_value = stats.ttest_ind(data1, data2)
            return {
                'test_type': 'independent_t_test',
                'statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'mean1': float(np.mean(data1)),
                'mean2': float(np.mean(data2))
            }
        elif test_type == 'mann_whitney':
            u_stat, p_value = stats.mannwhitneyu(data1, data2)
            return {
                'test_type': 'mann_whitney_u',
                'statistic': float(u_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        else:
            raise ValueError(f"Unknown test type: {test_type}")


class HypothesisTester:
    """Advanced hypothesis testing tools."""

    def __init__(self):
        """Initialize hypothesis tester."""
        pass

    def paired_t_test(self, before: Union[List, np.ndarray],
                     after: Union[List, np.ndarray]) -> Dict[str, Any]:
        """Perform paired t-test."""
        before = np.array(before)
        after = np.array(after)

        if len(before) != len(after):
            raise ValueError("Before and after data must have the same length")

        differences = after - before
        t_stat, p_value = stats.ttest_1samp(differences, 0)

        return {
            'test_type': 'paired_t_test',
            'statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'mean_difference': float(np.mean(differences)),
            'std_difference': float(np.std(differences, ddof=1)),
            'effect_size': float(np.mean(differences) / np.std(differences, ddof=1))
        }

    def anova_test(self, *groups) -> Dict[str, Any]:
        """Perform one-way ANOVA test."""
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            return {
                'test_type': 'anova',
                'statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'num_groups': len(groups),
                'group_sizes': [len(g) for g in groups]
            }
        except Exception as e:
            return {
                'test_type': 'anova',
                'error': str(e),
                'significant': False
            }

    def chi_square_test(self, observed: np.ndarray, expected: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Perform chi-square test."""
        if expected is None:
            # Goodness of fit test - assume uniform distribution
            expected = np.full_like(observed, np.sum(observed) / len(observed))

        chi2_stat, p_value = stats.chisquare(observed, expected)

        return {
            'test_type': 'chi_square',
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'degrees_of_freedom': len(observed) - 1
        }

    def regression_analysis(self, x: Union[List, np.ndarray],
                          y: Union[List, np.ndarray]) -> Dict[str, Any]:
        """Perform linear regression analysis."""
        x = np.array(x)
        y = np.array(y)

        # Add constant term for intercept
        X = np.column_stack([np.ones(len(x)), x])

        # Compute regression coefficients
        try:
            beta = np.linalg.inv(X.T @ X) @ X.T @ y
            y_pred = X @ beta
            residuals = y - y_pred

            # Compute statistics
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)

            # Standard errors
            mse = ss_res / (len(x) - 2)
            se = np.sqrt(np.diag(mse * np.linalg.inv(X.T @ X)))

            return {
                'intercept': float(beta[0]),
                'slope': float(beta[1]),
                'r_squared': float(r_squared),
                'intercept_std_error': float(se[0]),
                'slope_std_error': float(se[1]),
                'residuals': residuals.tolist(),
                'predicted': y_pred.tolist()
            }
        except Exception as e:
            return {
                'error': str(e),
                'intercept': None,
                'slope': None,
                'r_squared': None
            }
