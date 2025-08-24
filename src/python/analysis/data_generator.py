"""Data Generation Tools

Tools for generating mathematical and statistical datasets for testing
and demonstration purposes.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable
from pathlib import Path
import json
import csv
from datetime import datetime
import random


class MathematicalDataGenerator:
    """Generate mathematical and statistical datasets"""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the data generator"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.output_dir = Path("generated_data")
        self.output_dir.mkdir(exist_ok=True)

    def generate_polynomial_data(self, coefficients: List[float],
                               domain: Tuple[float, float],
                               noise_std: float = 0.0,
                               num_points: int = 100) -> Dict[str, Any]:
        """Generate data from a polynomial function with optional noise"""
        x = np.linspace(domain[0], domain[1], num_points)

        # Evaluate polynomial
        y_clean = np.zeros_like(x)
        for i, coeff in enumerate(coefficients):
            y_clean += coeff * (x ** i)

        # Add noise if specified
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, len(x))
            y = y_clean + noise
        else:
            y = y_clean

        data = {
            'x': x.tolist(),
            'y': y.tolist(),
            'y_clean': y_clean.tolist(),
            'coefficients': coefficients,
            'domain': domain,
            'noise_std': noise_std,
            'num_points': num_points,
            'polynomial_degree': len(coefficients) - 1
        }

        return data

    def generate_trigonometric_data(self, amplitude: float = 1.0,
                                   frequency: float = 1.0,
                                   phase: float = 0.0,
                                   domain: Tuple[float, float] = (0, 2*np.pi),
                                   noise_std: float = 0.0,
                                   num_points: int = 100) -> Dict[str, Any]:
        """Generate trigonometric function data"""
        x = np.linspace(domain[0], domain[1], num_points)

        # Generate sine wave
        y_clean = amplitude * np.sin(frequency * x + phase)

        # Add noise if specified
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, len(x))
            y = y_clean + noise
        else:
            y = y_clean

        data = {
            'x': x.tolist(),
            'y': y.tolist(),
            'y_clean': y_clean.tolist(),
            'amplitude': amplitude,
            'frequency': frequency,
            'phase': phase,
            'domain': domain,
            'noise_std': noise_std,
            'num_points': num_points
        }

        return data

    def generate_exponential_data(self, growth_rate: float = 0.1,
                                initial_value: float = 1.0,
                                domain: Tuple[float, float] = (0, 10),
                                noise_std: float = 0.0,
                                num_points: int = 100) -> Dict[str, Any]:
        """Generate exponential function data"""
        x = np.linspace(domain[0], domain[1], num_points)

        # Generate exponential function
        y_clean = initial_value * np.exp(growth_rate * x)

        # Add noise if specified
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, len(x))
            y = y_clean + noise
        else:
            y = y_clean

        data = {
            'x': x.tolist(),
            'y': y.tolist(),
            'y_clean': y_clean.tolist(),
            'growth_rate': growth_rate,
            'initial_value': initial_value,
            'domain': domain,
            'noise_std': noise_std,
            'num_points': num_points
        }

        return data

    def generate_statistical_data(self, distribution: str = 'normal',
                                parameters: Dict[str, float] = None,
                                sample_size: int = 1000) -> Dict[str, Any]:
        """Generate statistical data from various distributions"""
        if parameters is None:
            parameters = {}

        data = {
            'distribution': distribution,
            'parameters': parameters,
            'sample_size': sample_size,
            'samples': []
        }

        if distribution == 'normal':
            mean = parameters.get('mean', 0)
            std = parameters.get('std', 1)
            samples = np.random.normal(mean, std, sample_size)
            data['samples'] = samples.tolist()

        elif distribution == 'uniform':
            low = parameters.get('low', 0)
            high = parameters.get('high', 1)
            samples = np.random.uniform(low, high, sample_size)
            data['samples'] = samples.tolist()

        elif distribution == 'exponential':
            scale = parameters.get('scale', 1)
            samples = np.random.exponential(scale, sample_size)
            data['samples'] = samples.tolist()

        elif distribution == 'gamma':
            shape = parameters.get('shape', 2)
            scale = parameters.get('scale', 1)
            samples = np.random.gamma(shape, scale, sample_size)
            data['samples'] = samples.tolist()

        elif distribution == 'beta':
            alpha = parameters.get('alpha', 2)
            beta = parameters.get('beta', 2)
            samples = np.random.beta(alpha, beta, sample_size)
            data['samples'] = samples.tolist()

        elif distribution == 'poisson':
            lam = parameters.get('lambda', 3)
            samples = np.random.poisson(lam, sample_size)
            data['samples'] = samples.tolist()

        else:
            raise ValueError(f"Unsupported distribution: {distribution}")

        return data

    def generate_time_series_data(self, trend_type: str = 'linear',
                                seasonality: bool = True,
                                noise_std: float = 0.1,
                                length: int = 365) -> Dict[str, Any]:
        """Generate time series data"""
        t = np.arange(length)

        # Generate trend component
        if trend_type == 'linear':
            trend = 0.01 * t
        elif trend_type == 'exponential':
            trend = np.exp(0.001 * t) - 1
        elif trend_type == 'logistic':
            trend = 1 / (1 + np.exp(-0.01 * (t - length/2)))
        else:
            trend = np.zeros(length)

        # Generate seasonal component
        if seasonality:
            seasonal = np.sin(2 * np.pi * t / 52) + 0.5 * np.sin(2 * np.pi * t / 12)
        else:
            seasonal = np.zeros(length)

        # Generate noise
        noise = np.random.normal(0, noise_std, length)

        # Combine components
        y = trend + seasonal + noise

        data = {
            'time': t.tolist(),
            'values': y.tolist(),
            'trend': trend.tolist(),
            'seasonal': seasonal.tolist(),
            'noise': noise.tolist(),
            'trend_type': trend_type,
            'seasonality': seasonality,
            'noise_std': noise_std,
            'length': length
        }

        return data

    def generate_network_data(self, num_nodes: int = 20,
                            edge_probability: float = 0.3,
                            directed: bool = False) -> Dict[str, Any]:
        """Generate network/graph data"""
        if directed:
            # Generate directed adjacency matrix
            adjacency = np.random.binomial(1, edge_probability, (num_nodes, num_nodes))
            # Remove self-loops
            np.fill_diagonal(adjacency, 0)
        else:
            # Generate undirected adjacency matrix (symmetric)
            upper_triangle = np.random.binomial(1, edge_probability,
                                              (num_nodes, num_nodes))
            adjacency = np.triu(upper_triangle, 1)
            adjacency = adjacency + adjacency.T

        # Generate node positions (for visualization)
        positions = {}
        for i in range(num_nodes):
            positions[i] = {
                'x': float(np.cos(2 * np.pi * i / num_nodes)),
                'y': float(np.sin(2 * np.pi * i / num_nodes))
            }

        # Calculate network metrics
        degrees = np.sum(adjacency, axis=1).tolist()
        avg_degree = float(np.mean(degrees))

        data = {
            'num_nodes': num_nodes,
            'edge_probability': edge_probability,
            'directed': directed,
            'adjacency_matrix': adjacency.tolist(),
            'node_positions': positions,
            'degrees': degrees,
            'average_degree': avg_degree,
            'num_edges': int(np.sum(adjacency) / (2 if not directed else 1))
        }

        return data

    def save_data(self, data: Dict[str, Any], filename: str,
                 format: str = 'json') -> Path:
        """Save generated data to file"""
        filepath = self.output_dir / f"{filename}.{format}"

        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'csv':
            # Convert to DataFrame if possible
            if 'x' in data and 'y' in data:
                df = pd.DataFrame({'x': data['x'], 'y': data['y']})
            elif 'samples' in data:
                df = pd.DataFrame({'samples': data['samples']})
            elif 'time' in data and 'values' in data:
                df = pd.DataFrame({'time': data['time'], 'values': data['values']})
            else:
                df = pd.DataFrame(data)

            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return filepath

    def load_data(self, filepath: Path) -> Dict[str, Any]:
        """Load data from file"""
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
            return df.to_dict('list')
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")

    def generate_comprehensive_dataset(self, name: str = "comprehensive") -> Dict[str, Any]:
        """Generate a comprehensive dataset with multiple components"""
        dataset = {
            'name': name,
            'generated_at': datetime.now().isoformat(),
            'components': {}
        }

        # Generate various types of data
        dataset['components']['polynomial'] = self.generate_polynomial_data(
            [1, -2, 1], (-3, 3), noise_std=0.1
        )

        dataset['components']['trigonometric'] = self.generate_trigonometric_data(
            amplitude=2, frequency=0.5, noise_std=0.2
        )

        dataset['components']['exponential'] = self.generate_exponential_data(
            growth_rate=0.05, noise_std=0.1
        )

        dataset['components']['normal_distribution'] = self.generate_statistical_data(
            'normal', {'mean': 5, 'std': 2}, 1000
        )

        dataset['components']['time_series'] = self.generate_time_series_data(
            'linear', True, 0.2, 100
        )

        dataset['components']['network'] = self.generate_network_data(
            15, 0.3, False
        )

        return dataset


def create_data_gallery():
    """Create a gallery of generated datasets"""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel.fit("📊 LeanNiche Data Generation Gallery", style="bold magenta"))

    generator = MathematicalDataGenerator(seed=42)

    # Generate polynomial data
    console.print("\\n1. 📈 Polynomial Data", style="bold blue")
    poly_data = generator.generate_polynomial_data([1, -2, 1], (-3, 3), noise_std=0.1)
    generator.save_data(poly_data, "polynomial_data")
    console.print("   ✅ Generated polynomial dataset")
    console.print(f"   📊 {len(poly_data['x'])} data points")

    # Generate trigonometric data
    console.print("\\n2. 🌊 Trigonometric Data", style="bold blue")
    trig_data = generator.generate_trigonometric_data(2, 0.5, noise_std=0.2)
    generator.save_data(trig_data, "trigonometric_data")
    console.print("   ✅ Generated trigonometric dataset")
    console.print(f"   📊 {len(trig_data['x'])} data points")

    # Generate statistical data
    console.print("\\n3. 📊 Statistical Data", style="bold blue")
    stat_data = generator.generate_statistical_data('normal', {'mean': 5, 'std': 2}, 1000)
    generator.save_data(stat_data, "statistical_data")
    console.print("   ✅ Generated statistical dataset")
    console.print(f"   📊 {len(stat_data['samples'])} samples")

    # Generate time series data
    console.print("\\n4. ⏱️  Time Series Data", style="bold blue")
    ts_data = generator.generate_time_series_data('linear', True, 0.2, 100)
    generator.save_data(ts_data, "time_series_data")
    console.print("   ✅ Generated time series dataset")
    console.print(f"   📊 {len(ts_data['values'])} time points")

    # Generate network data
    console.print("\\n5. 🔗 Network Data", style="bold blue")
    network_data = generator.generate_network_data(15, 0.3, False)
    generator.save_data(network_data, "network_data")
    console.print("   ✅ Generated network dataset")
    console.print(f"   📊 {network_data['num_nodes']} nodes, {network_data['num_edges']} edges")

    # Generate comprehensive dataset
    console.print("\\n6. 🌟 Comprehensive Dataset", style="bold blue")
    comprehensive_data = generator.generate_comprehensive_dataset()
    generator.save_data(comprehensive_data, "comprehensive_dataset")
    console.print("   ✅ Generated comprehensive dataset")
    console.print(f"   📊 {len(comprehensive_data['components'])} components")

    console.print("\\n🎉 Data Generation Gallery Complete!")
    console.print(f"   📁 All datasets saved to: {generator.output_dir}")
    console.print("\\n📋 Generated Files:")
    for file in generator.output_dir.glob("*.json"):
        console.print(f"   📄 {file.name}")


if __name__ == "__main__":
    create_data_gallery()
