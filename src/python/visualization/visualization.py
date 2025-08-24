"""Mathematical Visualization Tools

Advanced visualization tools for mathematical concepts, data analysis,
and research visualization.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from scipy import stats
import networkx as nx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json
import os
from pathlib import Path

console = Console()

class MathematicalVisualizer:
    """Main class for mathematical visualization"""

    def __init__(self, style: str = "seaborn-v0_8"):
        """Initialize the visualizer with a specific style"""
        plt.style.use(style)
        sns.set_palette("husl")
        self.output_dir = Path("visualizations")
        self.output_dir.mkdir(exist_ok=True)

    def plot_function(self, func: callable, domain: Tuple[float, float],
                     title: str = "Function Plot", save_path: Optional[str] = None,
                     **kwargs) -> plt.Figure:
        """Plot a mathematical function"""
        x = np.linspace(domain[0], domain[1], 1000)
        y = func(x)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, y, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_trajectory(self, trajectory: List[Tuple[float, float]],
                       title: str = "Phase Portrait", save_path: Optional[str] = None,
                       **kwargs) -> plt.Figure:
        """Plot a dynamical system trajectory"""
        if not trajectory:
            raise ValueError("Empty trajectory")

        x_coords, y_coords = zip(*trajectory)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x_coords, y_coords, marker='o', markersize=2, **kwargs)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', 'datalim')

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_statistical_data(self, data: List[float], title: str = "Statistical Analysis",
                            save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive statistical visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)

        # Histogram
        axes[0, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title("Histogram")
        axes[0, 0].set_xlabel("Value")
        axes[0, 0].set_ylabel("Frequency")

        # Box plot
        axes[0, 1].boxplot(data)
        axes[0, 1].set_title("Box Plot")
        axes[0, 1].set_ylabel("Value")

        # Q-Q plot
        stats.probplot(data, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title("Q-Q Plot")

        # Cumulative distribution
        sorted_data = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        axes[1, 1].plot(sorted_data, y, marker='.', linestyle='none')
        axes[1, 1].set_title("Cumulative Distribution")
        axes[1, 1].set_xlabel("Value")
        axes[1, 1].set_ylabel("Cumulative Probability")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_interactive_plot(self, x_data: List[float], y_data: List[float],
                              plot_type: str = "scatter", title: str = "Interactive Plot",
                              save_path: Optional[str] = None) -> go.Figure:
        """Create interactive plotly visualization"""
        if plot_type == "scatter":
            fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='markers'))
        elif plot_type == "line":
            fig = go.Figure(data=go.Scatter(x=x_data, y=y_data, mode='lines'))
        elif plot_type == "bar":
            fig = go.Figure(data=go.Bar(x=x_data, y=y_data))
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

        fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            template="plotly_white"
        )

        if save_path:
            fig.write_html(str(self.output_dir / f"{save_path}.html"))
            fig.write_image(str(self.output_dir / f"{save_path}.png"))

        return fig

    def visualize_network(self, adjacency_matrix: np.ndarray,
                         title: str = "Network Visualization",
                         save_path: Optional[str] = None) -> plt.Figure:
        """Visualize a network/graph"""
        G = nx.from_numpy_array(adjacency_matrix)

        fig, ax = plt.subplots(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)

        nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue',
                node_size=500, font_size=10, font_weight='bold',
                edge_color='gray', width=2, alpha=0.7)

        ax.set_title(title)

        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

class StatisticalAnalyzer:
    """Statistical analysis and visualization tools"""

    def __init__(self):
        self.visualizer = MathematicalVisualizer()

    def analyze_dataset(self, data: List[float], name: str = "Dataset") -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        analysis = {
            'name': name,
            'n': len(data),
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'var': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'range': np.max(data) - np.min(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'iqr': np.percentile(data, 75) - np.percentile(data, 25),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }

        # Normality tests
        if len(data) >= 3:
            analysis['shapiro_test'] = stats.shapiro(data)
            analysis['normaltest'] = stats.normaltest(data)

        return analysis

    def create_analysis_report(self, data: List[float], name: str = "Dataset",
                             save_path: Optional[str] = None) -> str:
        """Create a formatted analysis report"""
        analysis = self.analyze_dataset(data, name)

        # Create visualization
        self.visualizer.plot_statistical_data(data, f"{name}_analysis.png")

        # Generate report
        report = f"""
        📊 Statistical Analysis Report: {name}
        {'=' * 50}

        📈 Basic Statistics:
        • Sample Size: {analysis['n']}
        • Mean: {analysis['mean']:.4f}
        • Median: {analysis['median']:.4f}
        • Standard Deviation: {analysis['std']:.4f}
        • Variance: {analysis['var']:.4f}
        • Range: {analysis['range']:.4f}
        • IQR: {analysis['iqr']:.4f}

        📊 Distribution Characteristics:
        • Skewness: {analysis['skewness']:.4f}
        • Kurtosis: {analysis['kurtosis']:.4f}
        • Min: {analysis['min']:.4f}
        • Max: {analysis['max']:.4f}
        • Q25: {analysis['q25']:.4f}
        • Q75: {analysis['q75']:.4f}
        """

        if 'shapiro_test' in analysis:
            report += f"""
        🧪 Normality Tests:
        • Shapiro-Wilk Test: p-value = {analysis['shapiro_test'].pvalue:.4f}
        • D'Agostino Test: p-value = {analysis['normaltest'].pvalue:.4f}
            """

        if save_path:
            with open(self.visualizer.output_dir / save_path, 'w') as f:
                f.write(report)

        return report

class DynamicalSystemsVisualizer:
    """Specialized visualization for dynamical systems"""

    def __init__(self):
        self.visualizer = MathematicalVisualizer()

    def plot_phase_portrait(self, vector_field_x: callable, vector_field_y: callable,
                          domain: Tuple[Tuple[float, float], Tuple[float, float]],
                          title: str = "Phase Portrait", save_path: Optional[str] = None,
                          trajectory_points: Optional[List[Tuple[float, float]]] = None) -> plt.Figure:
        """Plot phase portrait of a 2D dynamical system"""
        x_min, x_max = domain[0]
        y_min, y_max = domain[1]

        x = np.linspace(x_min, x_max, 20)
        y = np.linspace(y_min, y_max, 20)
        X, Y = np.meshgrid(x, y)

        U = vector_field_x(X, Y)
        V = vector_field_y(X, Y)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot vector field
        ax.quiver(X, Y, U, V, alpha=0.7)

        # Plot trajectories if provided
        if trajectory_points:
            x_traj, y_traj = zip(*trajectory_points)
            ax.plot(x_traj, y_traj, 'r-', linewidth=2, label='Trajectory')

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        ax.legend()

        if save_path:
            plt.savefig(self.visualizer.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

    def plot_bifurcation_diagram(self, parameter_range: Tuple[float, float],
                                system_function: callable, title: str = "Bifurcation Diagram",
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot bifurcation diagram for a parameterized system"""
        param_min, param_max = parameter_range
        params = np.linspace(param_min, param_max, 1000)

        fig, ax = plt.subplots(figsize=(12, 8))

        for param in params[::10]:  # Sample parameters
            try:
                # Simulate system for this parameter
                x0 = 0.1
                trajectory = [x0]
                for _ in range(1000):
                    x_new = system_function(trajectory[-1], param)
                    trajectory.append(x_new)

                # Plot last 100 points (attractors)
                attractors = trajectory[-100:]
                ax.scatter([param] * len(attractors), attractors, s=1, c='black', alpha=0.7)
            except:
                continue

        ax.set_title(title)
        ax.set_xlabel("Parameter")
        ax.set_ylabel("x")
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(self.visualizer.output_dir / save_path, dpi=300, bbox_inches='tight')

        return fig

def create_visualization_gallery():
    """Create a comprehensive visualization gallery"""
    console.print(Panel.fit("🎨 LeanNiche Visualization Gallery", style="bold magenta"))

    # Initialize visualizers
    viz = MathematicalVisualizer()
    stat_analyzer = StatisticalAnalyzer()
    dyn_viz = DynamicalSystemsVisualizer()

    # Example 1: Function plot
    console.print("\n1. 📈 Function Visualization", style="bold blue")
    def quadratic(x):
        return x**2 - 2*x + 1

    fig1 = viz.plot_function(quadratic, (-3, 5), "Quadratic Function f(x) = x² - 2x + 1",
                           "quadratic_function.png")
    console.print("   ✅ Created quadratic function plot")

    # Example 2: Statistical analysis
    console.print("\n2. 📊 Statistical Data Analysis", style="bold blue")
    np.random.seed(42)
    data = np.random.normal(5, 2, 1000).tolist()
    analysis = stat_analyzer.analyze_dataset(data, "Normal Distribution Sample")
    report = stat_analyzer.create_analysis_report(data, "normal_sample_analysis.txt")

    console.print(f"   📊 Sample Statistics:")
    console.print(f"      • Sample Size: {analysis['n']}")
    console.print(f"      • Mean: {analysis['mean']:.3f}")
    console.print(f"      • Standard Deviation: {analysis['std']:.3f}")
    console.print(f"      • Skewness: {analysis['skewness']:.3f}")

    # Example 3: Dynamical system
    console.print("\n3. 🔄 Dynamical System Visualization", style="bold blue")
    def logistic_map(x, r):
        return r * x * (1 - x)

    fig3 = dyn_viz.plot_bifurcation_diagram((2.5, 4.0), logistic_map,
                                          "Logistic Map Bifurcation Diagram",
                                          "logistic_bifurcation.png")
    console.print("   ✅ Created logistic map bifurcation diagram")

    # Example 4: Interactive plot
    console.print("\n4. 🌐 Interactive Visualization", style="bold blue")
    x_data = np.linspace(0, 4*np.pi, 100)
    y_data = np.sin(x_data) * np.exp(-x_data/10)

    fig4 = viz.create_interactive_plot(x_data.tolist(), y_data.tolist(),
                                     plot_type="line", title="Damped Sine Wave",
                                     save_path="damped_sine_wave")
    console.print("   ✅ Created interactive damped sine wave plot")

    console.print("\n🎉 Visualization Gallery Complete!", style="bold green")
    console.print(f"   📁 All visualizations saved to: {viz.output_dir}")

if __name__ == "__main__":
    create_visualization_gallery()
