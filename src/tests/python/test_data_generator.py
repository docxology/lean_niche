"""Unit tests for MathematicalDataGenerator"""

import numpy as np
from src.python.analysis.data_generator import MathematicalDataGenerator


def test_generate_polynomial_data_shape():
    gen = MathematicalDataGenerator(seed=0)
    data = gen.generate_polynomial_data([1, 0, -1], (-1, 1), noise_std=0.0, num_points=50)

    assert 'x' in data and 'y' in data
    assert len(data['x']) == 50
    assert len(data['y']) == 50


def test_generate_trigonometric_data_properties():
    gen = MathematicalDataGenerator(seed=1)
    data = gen.generate_trigonometric_data(amplitude=2.0, frequency=0.5, num_points=100)

    assert abs(max(data['y']) - 2.0) <= 2.0
    assert data['num_points'] == 100


def test_save_and_load_json(tmp_path):
    gen = MathematicalDataGenerator(seed=2)
    data = gen.generate_statistical_data('normal', {'mean': 0, 'std': 1}, sample_size=10)

    path = gen.save_data(data, filename=str(tmp_path / 'test_stats'), format='json')
    loaded = gen.load_data(path)

    assert 'samples' in loaded
    assert len(loaded['samples']) == 10


