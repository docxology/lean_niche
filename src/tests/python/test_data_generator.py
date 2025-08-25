"""Comprehensive unit tests for MathematicalDataGenerator"""

import numpy as np
import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.python.analysis.data_generator import MathematicalDataGenerator


class TestMathematicalDataGenerator:
    """Comprehensive tests for MathematicalDataGenerator"""

    def setup_method(self):
        """Setup test fixtures"""
        self.generator = MathematicalDataGenerator(seed=42)

    def test_initialization(self):
        """Test data generator initialization"""
        assert isinstance(self.generator, MathematicalDataGenerator)
        assert hasattr(self.generator, 'generate_polynomial_data')
        assert hasattr(self.generator, 'generate_trigonometric_data')
        assert hasattr(self.generator, 'generate_exponential_data')
        assert hasattr(self.generator, 'generate_statistical_data')
        assert hasattr(self.generator, 'generate_time_series_data')
        assert hasattr(self.generator, 'generate_network_data')
        assert hasattr(self.generator, 'save_data')
        assert hasattr(self.generator, 'load_data')

    def test_initialization_with_seed(self):
        """Test initialization with specific seed"""
        gen = MathematicalDataGenerator(seed=123)
        assert gen.seed == 123

    def test_initialization_without_seed(self):
        """Test initialization without seed"""
        gen = MathematicalDataGenerator()
        assert gen.seed is None

    def test_generate_polynomial_data_shape(self):
        """Test polynomial data generation shape"""
        data = self.generator.generate_polynomial_data([1, 0, -1], (-1, 1), noise_std=0.0, num_points=50)

        assert 'x' in data and 'y' in data
        assert len(data['x']) == 50
        assert len(data['y']) == 50
        assert 'coefficients' in data
        assert 'domain' in data
        assert 'noise_std' in data
        assert data['coefficients'] == [1, 0, -1]

    def test_generate_polynomial_data_linear(self):
        """Test polynomial data generation for linear function"""
        data = self.generator.generate_polynomial_data([2, 3], (-2, 2), noise_std=0.0, num_points=10)

        # y = 2x + 3
        expected_y = [2*x + 3 for x in data['x']]
        assert data['y'] == expected_y

    def test_generate_polynomial_data_quadratic(self):
        """Test polynomial data generation for quadratic function"""
        data = self.generator.generate_polynomial_data([1, 0, 1], (-2, 2), noise_std=0.0, num_points=5)

        # y = x^2 + 1
        expected_y = [x**2 + 1 for x in data['x']]
        assert data['y'] == expected_y

    def test_generate_polynomial_data_with_noise(self):
        """Test polynomial data generation with noise"""
        noise_std = 0.1
        data = self.generator.generate_polynomial_data([1, 0], (-1, 1), noise_std=noise_std, num_points=100)

        assert len(data['x']) == 100
        assert len(data['y']) == 100
        assert data['noise_std'] == noise_std

        # Check that y values are close to x (linear function with noise)
        y_diff = [abs(y - x) for x, y in zip(data['x'], data['y'])]
        assert all(diff < 3 * noise_std for diff in y_diff)  # 99.7% of values within 3σ

    def test_generate_polynomial_data_invalid_coefficients(self):
        """Test polynomial data generation with invalid coefficients"""
        with pytest.raises(ValueError):
            self.generator.generate_polynomial_data([], (-1, 1))

    def test_generate_polynomial_data_invalid_num_points(self):
        """Test polynomial data generation with invalid number of points"""
        with pytest.raises(ValueError):
            self.generator.generate_polynomial_data([1, 0, -1], (-1, 1), num_points=0)

    def test_generate_trigonometric_data_properties(self):
        """Test trigonometric data generation properties"""
        amplitude = 2.0
        frequency = 0.5
        data = self.generator.generate_trigonometric_data(
            amplitude=amplitude,
            frequency=frequency,
            domain=(0, 2*np.pi),
            noise_std=0.0,
            num_points=100
        )

        assert abs(max(data['y']) - amplitude) <= amplitude * 0.01  # Allow small numerical error
        assert data['num_points'] == 100
        assert 'amplitude' in data
        assert 'frequency' in data
        assert 'phase' in data
        assert data['amplitude'] == amplitude
        assert data['frequency'] == frequency

    def test_generate_trigonometric_data_sine(self):
        """Test sine wave generation"""
        data = self.generator.generate_trigonometric_data(
            amplitude=1.0,
            frequency=1.0,
            phase=0.0,
            domain=(0, np.pi),
            noise_std=0.0,
            num_points=3
        )

        # At x=0, π/2, π
        expected_y = [0, 1, 0]  # sin(0), sin(π/2), sin(π)
        assert len(data['y']) == 3
        for i, expected in enumerate(expected_y):
            assert abs(data['y'][i] - expected) < 1e-6

    def test_generate_trigonometric_data_with_phase(self):
        """Test trigonometric data generation with phase shift"""
        phase = np.pi/4
        data = self.generator.generate_trigonometric_data(
            amplitude=1.0,
            frequency=1.0,
            phase=phase,
            domain=(0, 0),
            noise_std=0.0,
            num_points=1
        )

        expected_y = np.sin(phase)
        assert abs(data['y'][0] - expected_y) < 1e-6

    def test_generate_trigonometric_data_with_noise(self):
        """Test trigonometric data generation with noise"""
        noise_std = 0.1
        data = self.generator.generate_trigonometric_data(
            amplitude=1.0,
            frequency=1.0,
            noise_std=noise_std,
            num_points=100
        )

        assert data['noise_std'] == noise_std
        # Check that signal is still roughly sinusoidal
        assert max(data['y']) > 0.5  # Should still reach high values
        assert min(data['y']) < -0.5  # Should still reach low values

    def test_generate_exponential_data_shape(self):
        """Test exponential data generation shape"""
        growth_rate = 0.1
        initial_value = 2.0
        data = self.generator.generate_exponential_data(
            growth_rate=growth_rate,
            initial_value=initial_value,
            domain=(0, 5),
            noise_std=0.0,
            num_points=10
        )

        assert len(data['x']) == 10
        assert len(data['y']) == 10
        assert data['y'][0] == initial_value  # First point should be initial value

    def test_generate_exponential_data_growth(self):
        """Test exponential growth pattern"""
        growth_rate = 0.5
        data = self.generator.generate_exponential_data(
            growth_rate=growth_rate,
            initial_value=1.0,
            domain=(0, 2),
            noise_std=0.0,
            num_points=3
        )

        # y = e^(growth_rate * x)
        expected_y = [np.exp(growth_rate * x) for x in data['x']]
        for actual, expected in zip(data['y'], expected_y):
            assert abs(actual - expected) < 1e-6

    def test_generate_exponential_data_decay(self):
        """Test exponential decay pattern"""
        growth_rate = -0.5
        data = self.generator.generate_exponential_data(
            growth_rate=growth_rate,
            initial_value=2.0,
            domain=(0, 2),
            noise_std=0.0,
            num_points=3
        )

        # Check that values are decreasing
        assert data['y'][0] > data['y'][1] > data['y'][2]

    def test_generate_exponential_data_with_noise(self):
        """Test exponential data generation with noise"""
        noise_std = 0.1
        data = self.generator.generate_exponential_data(
            growth_rate=0.1,
            initial_value=1.0,
            domain=(0, 5),
            noise_std=noise_std,
            num_points=100
        )

        assert data['noise_std'] == noise_std
        # Check that overall trend is still increasing
        assert data['y'][-1] > data['y'][0]

    def test_generate_statistical_data_normal(self):
        """Test statistical data generation for normal distribution"""
        params = {'mean': 5.0, 'std': 2.0}
        data = self.generator.generate_statistical_data('normal', params, sample_size=1000)

        assert 'samples' in data
        assert len(data['samples']) == 1000
        assert 'distribution' in data
        assert 'parameters' in data
        assert data['distribution'] == 'normal'
        assert data['parameters'] == params

        # Check statistical properties
        samples = data['samples']
        sample_mean = sum(samples) / len(samples)
        sample_std = np.std(samples)
        assert abs(sample_mean - params['mean']) < 0.1  # Should be close to true mean
        assert abs(sample_std - params['std']) < 0.1    # Should be close to true std

    def test_generate_statistical_data_uniform(self):
        """Test statistical data generation for uniform distribution"""
        params = {'low': 0.0, 'high': 10.0}
        data = self.generator.generate_statistical_data('uniform', params, sample_size=1000)

        assert len(data['samples']) == 1000
        assert data['distribution'] == 'uniform'
        assert data['parameters'] == params

        samples = data['samples']
        assert all(0.0 <= s <= 10.0 for s in samples)

    def test_generate_statistical_data_exponential(self):
        """Test statistical data generation for exponential distribution"""
        params = {'scale': 2.0}
        data = self.generator.generate_statistical_data('exponential', params, sample_size=1000)

        assert len(data['samples']) == 1000
        assert data['distribution'] == 'exponential'
        assert data['parameters'] == params

        samples = data['samples']
        assert all(s >= 0 for s in samples)

    def test_generate_statistical_data_invalid_distribution(self):
        """Test statistical data generation with invalid distribution"""
        with pytest.raises(ValueError):
            self.generator.generate_statistical_data('invalid', {}, sample_size=10)

    def test_generate_statistical_data_small_sample(self):
        """Test statistical data generation with small sample"""
        data = self.generator.generate_statistical_data('normal', {'mean': 0, 'std': 1}, sample_size=5)

        assert len(data['samples']) == 5
        assert all(isinstance(s, (int, float)) for s in data['samples'])

    def test_generate_time_series_data_linear_trend(self):
        """Test time series data generation with linear trend"""
        data = self.generator.generate_time_series_data(
            trend_type='linear',
            seasonality=False,
            noise_std=0.0,
            length=10
        )

        assert len(data['values']) == 10
        assert 'trend_type' in data
        assert 'seasonality' in data
        assert data['trend_type'] == 'linear'
        assert data['seasonality'] == False

        # Check linear trend
        values = data['values']
        assert values[-1] > values[0]  # Should be increasing

    def test_generate_time_series_data_with_seasonality(self):
        """Test time series data generation with seasonality"""
        data = self.generator.generate_time_series_data(
            trend_type='constant',
            seasonality=True,
            noise_std=0.0,
            length=20
        )

        assert len(data['values']) == 20
        assert data['seasonality'] == True

        # Check for periodic pattern
        values = data['values']
        # Values should repeat every period (assuming period=7 for weekly pattern)
        if len(values) >= 14:  # Need at least 2 periods
            assert abs(values[0] - values[7]) < 1e-6  # Should repeat weekly
            assert abs(values[1] - values[8]) < 1e-6

    def test_generate_time_series_data_with_noise(self):
        """Test time series data generation with noise"""
        noise_std = 0.5
        data = self.generator.generate_time_series_data(
            trend_type='constant',
            seasonality=False,
            noise_std=noise_std,
            length=100
        )

        assert data['noise_std'] == noise_std
        values = data['values']
        # Check that values vary around the mean
        mean_val = sum(values) / len(values)
        deviations = [abs(v - mean_val) for v in values]
        assert any(d > noise_std for d in deviations)  # Some values should deviate

    def test_generate_network_data_undirected(self):
        """Test network data generation for undirected graphs"""
        num_nodes = 10
        edge_probability = 0.3
        data = self.generator.generate_network_data(
            num_nodes=num_nodes,
            edge_probability=edge_probability,
            directed=False
        )

        assert 'adjacency_matrix' in data
        assert 'num_nodes' in data
        assert 'edge_probability' in data
        assert 'directed' in data
        assert data['num_nodes'] == num_nodes
        assert data['edge_probability'] == edge_probability
        assert data['directed'] == False

        matrix = data['adjacency_matrix']
        assert len(matrix) == num_nodes
        assert all(len(row) == num_nodes for row in matrix)
        assert all(matrix[i][j] == matrix[j][i] for i in range(num_nodes) for j in range(num_nodes))

    def test_generate_network_data_directed(self):
        """Test network data generation for directed graphs"""
        num_nodes = 8
        data = self.generator.generate_network_data(
            num_nodes=num_nodes,
            edge_probability=0.2,
            directed=True
        )

        assert data['directed'] == True
        matrix = data['adjacency_matrix']
        assert len(matrix) == num_nodes

        # Check that matrix is not necessarily symmetric
        is_symmetric = all(matrix[i][j] == matrix[j][i] for i in range(num_nodes) for j in range(num_nodes))
        # Should not necessarily be symmetric for directed graphs
        assert isinstance(is_symmetric, bool)  # Just check it's a boolean

    def test_generate_network_data_invalid_nodes(self):
        """Test network data generation with invalid number of nodes"""
        with pytest.raises(ValueError):
            self.generator.generate_network_data(num_nodes=0)

    def test_save_and_load_json(self):
        """Test saving and loading data in JSON format"""
        data = self.generator.generate_statistical_data('normal', {'mean': 0, 'std': 1}, sample_size=10)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self.generator.save_data(data, filename=tmp_dir + '/test_stats', format='json')
            loaded = self.generator.load_data(path)

            assert 'samples' in loaded
            assert len(loaded['samples']) == 10
            assert loaded['samples'] == data['samples']

    def test_save_data_csv(self):
        """Test saving data in CSV format"""
        data = self.generator.generate_polynomial_data([1, 0, -1], (-1, 1), num_points=5)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self.generator.save_data(data, filename=tmp_dir + '/test_data', format='csv')
            assert path.exists()
            assert path.suffix == '.csv'

    def test_save_data_invalid_format(self):
        """Test saving data with invalid format"""
        data = {'test': 'data'}

        with pytest.raises(ValueError):
            self.generator.save_data(data, filename='test', format='invalid')

    def test_load_data_invalid_file(self):
        """Test loading data from invalid file"""
        with pytest.raises(FileNotFoundError):
            self.generator.load_data(Path('/nonexistent/file.json'))

    def test_generate_comprehensive_dataset(self):
        """Test comprehensive dataset generation"""
        data = self.generator.generate_comprehensive_dataset(name="test_comprehensive")

        assert 'name' in data
        assert 'timestamp' in data
        assert 'datasets' in data
        assert data['name'] == "test_comprehensive"
        assert isinstance(data['datasets'], dict)

        # Check that multiple types of data are included
        datasets = data['datasets']
        assert 'polynomial' in datasets
        assert 'trigonometric' in datasets
        assert 'statistical' in datasets

    def test_reproducibility_with_seed(self):
        """Test reproducibility when using seed"""
        gen1 = MathematicalDataGenerator(seed=123)
        gen2 = MathematicalDataGenerator(seed=123)

        data1 = gen1.generate_polynomial_data([1, 2, 3], (-1, 1), num_points=5)
        data2 = gen2.generate_polynomial_data([1, 2, 3], (-1, 1), num_points=5)

        assert data1['x'] == data2['x']
        assert data1['y'] == data2['y']

    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data"""
        gen1 = MathematicalDataGenerator(seed=123)
        gen2 = MathematicalDataGenerator(seed=456)

        data1 = gen1.generate_statistical_data('normal', {'mean': 0, 'std': 1}, sample_size=10)
        data2 = gen2.generate_statistical_data('normal', {'mean': 0, 'std': 1}, sample_size=10)

        assert data1['samples'] != data2['samples']

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases"""
        # Test with very small domain
        data = self.generator.generate_polynomial_data([1], (0, 0.001), num_points=1)
        assert len(data['x']) == 1
        assert len(data['y']) == 1

        # Test with large number of points
        data = self.generator.generate_trigonometric_data(num_points=10000)
        assert len(data['x']) == 10000
        assert len(data['y']) == 10000

    def test_data_structure_consistency(self):
        """Test that all generated data has consistent structure"""
        # Test polynomial data structure
        poly_data = self.generator.generate_polynomial_data([1, 2], (-1, 1), num_points=5)
        required_keys = ['x', 'y', 'coefficients', 'domain', 'noise_std']
        for key in required_keys:
            assert key in poly_data

        # Test trigonometric data structure
        trig_data = self.generator.generate_trigonometric_data(num_points=5)
        required_keys = ['x', 'y', 'amplitude', 'frequency', 'phase', 'domain', 'noise_std', 'num_points']
        for key in required_keys:
            assert key in trig_data

        # Test statistical data structure
        stat_data = self.generator.generate_statistical_data('normal', {'mean': 0, 'std': 1}, sample_size=5)
        required_keys = ['samples', 'distribution', 'parameters']
        for key in required_keys:
            assert key in stat_data


