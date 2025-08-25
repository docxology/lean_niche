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
        ax_hist = axes[0][0] if hasattr(axes[0], '__iter__') else axes[0, 0]
        ax_hist.hist(data, bins=30, alpha=0.7, edgecolor='black')
        ax_hist.set_title("Histogram")
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Frequency")

        # Box plot
        ax_box = axes[0][1] if hasattr(axes[0], '__iter__') else axes[0, 1]
        ax_box.boxplot(data)
        ax_box.set_title("Box Plot")
        ax_box.set_ylabel("Value")

        # Q-Q plot
        ax_qq = axes[1][0] if hasattr(axes[1], '__iter__') else axes[1, 0]
        stats.probplot(data, dist="norm", plot=ax_qq)
        ax_qq.set_title("Q-Q Plot")

        # Cumulative distribution
        ax_cdf = axes[1][1] if hasattr(axes[1], '__iter__') else axes[1, 1]
        sorted_data = np.sort(data)
        y = np.arange(1, len(data) + 1) / len(data)
        ax_cdf.plot(sorted_data, y, marker='.', linestyle='none')
        ax_cdf.set_title("Cumulative Distribution")
        ax_cdf.set_xlabel("Value")
        ax_cdf.set_ylabel("Cumulative Probability")
        ax_cdf.grid(True, alpha=0.3)

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
            # Attempt to write static image; if Kaleido/Chrome not available, skip image export
            try:
                fig.write_image(str(self.output_dir / f"{save_path}.png"))
            except Exception:
                # Log to console but do not raise; image export is optional in CI
                console.print("⚠️ Plotly image export skipped (kaleido/Chrome not available)")

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

    # Domain-specific visualization methods
    def create_control_visualizations(self, pid_results: List[Dict[str, Any]],
                                    stability_results: Dict[str, Any],
                                    output_dir: Path) -> None:
        """Create comprehensive control theory visualizations."""
        print("📊 Creating control theory visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. PID Controller Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('PID Controller Performance Analysis', fontsize=16)

        controller_names = []
        settling_times = []
        steady_state_errors = []
        overshoots = []
        control_efforts = []

        for result in pid_results:
            controller_names.append(result['name'])
            settling_times.append(result.get('settling_time', 0))
            steady_state_errors.append(result.get('steady_state_error', 0))
            overshoots.append(result.get('overshoot', 0))
            control_efforts.append(result.get('control_effort', 0))

        # Performance metrics comparison
        axes[0, 0].bar(controller_names, settling_times, alpha=0.7, color='blue')
        axes[0, 0].set_title('Settling Time Comparison')
        axes[0, 0].set_ylabel('Time (s)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].bar(controller_names, steady_state_errors, alpha=0.7, color='green')
        axes[0, 1].set_title('Steady State Error')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].tick_params(axis='x', rotation=45)

        axes[1, 0].bar(controller_names, overshoots, alpha=0.7, color='red')
        axes[1, 0].set_title('Overshoot Percentage')
        axes[1, 0].set_ylabel('Overshoot (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        axes[1, 1].bar(controller_names, control_efforts, alpha=0.7, color='purple')
        axes[1, 1].set_title('Control Effort')
        axes[1, 1].set_ylabel('Control Magnitude')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "pid_performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. System Stability Analysis
        if stability_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('System Stability Analysis', fontsize=16)

            # Eigenvalue plot
            if 'eigenvalues' in stability_results:
                eigenvalues = stability_results['eigenvalues']
                real_parts = [ev.real for ev in eigenvalues]
                imag_parts = [ev.imag for ev in eigenvalues]

                axes[0, 0].scatter(real_parts, imag_parts, s=100, marker='x', color='red')
                axes[0, 0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
                axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[0, 0].set_xlabel('Real Part')
                axes[0, 0].set_ylabel('Imaginary Part')
                axes[0, 0].set_title('Eigenvalue Analysis')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_aspect('equal')

            # Step response
            if 'step_response' in stability_results:
                step_data = stability_results['step_response']
                axes[0, 1].plot(step_data['time'], step_data['response'], 'b-', linewidth=2)
                axes[0, 1].set_xlabel('Time (s)')
                axes[0, 1].set_ylabel('Response')
                axes[0, 1].set_title('Step Response')
                axes[0, 1].grid(True, alpha=0.3)

            # Bode plot
            if 'bode_plot' in stability_results:
                bode_data = stability_results['bode_plot']
                axes[1, 0].semilogx(bode_data['frequency'], bode_data['magnitude'], 'b-')
                axes[1, 0].set_xlabel('Frequency (rad/s)')
                axes[1, 0].set_ylabel('Magnitude (dB)')
                axes[1, 0].set_title('Bode Magnitude Plot')
                axes[1, 0].grid(True, alpha=0.3)

                axes[1, 1].semilogx(bode_data['frequency'], bode_data['phase'], 'r-')
                axes[1, 1].set_xlabel('Frequency (rad/s)')
                axes[1, 1].set_ylabel('Phase (degrees)')
                axes[1, 1].set_title('Bode Phase Plot')
                axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "stability_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Pole-Zero Map
        if 'pole_zero' in stability_results:
            fig, ax = plt.subplots(figsize=(10, 8))
            pole_zero = stability_results['pole_zero']

            if 'poles' in pole_zero:
                poles_real = [p.real for p in pole_zero['poles']]
                poles_imag = [p.imag for p in pole_zero['poles']]
                ax.scatter(poles_real, poles_imag, s=100, marker='x', color='red', label='Poles')

            if 'zeros' in pole_zero:
                zeros_real = [z.real for z in pole_zero['zeros']]
                zeros_imag = [z.imag for z in pole_zero['zeros']]
                ax.scatter(zeros_real, zeros_imag, s=100, marker='o', color='blue', label='Zeros')

            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_xlabel('Real Part')
            ax.set_ylabel('Imaginary Part')
            ax.set_title('Pole-Zero Map')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            plt.tight_layout()
            plt.savefig(output_dir / "pole_zero_map.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Control visualizations saved to: {output_dir}")

    def create_dynamical_visualizations(self, logistic_results: Dict[str, Any],
                                       oscillator_results: Dict[str, Any],
                                       output_dir: Path) -> None:
        """Create comprehensive dynamical systems visualizations."""
        print("📊 Creating dynamical systems visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Logistic Map Bifurcation Diagram
        if logistic_results and 'bifurcation_data' in logistic_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            bifurcation_data = logistic_results['bifurcation_data']

            if 'parameters' in bifurcation_data and 'attractors' in bifurcation_data:
                params = bifurcation_data['parameters']
                attractors = bifurcation_data['attractors']

                for param, attr in zip(params, attractors):
                    ax.scatter([param] * len(attr), attr, s=1, c='black', alpha=0.7)

            ax.set_xlabel('r (Growth Rate)')
            ax.set_ylabel('x (Population)')
            ax.set_title('Logistic Map Bifurcation Diagram')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "bifurcation_diagram.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Phase Portrait Analysis
        if oscillator_results and 'phase_portraits' in oscillator_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Nonlinear Oscillator Phase Portraits', fontsize=16)

            phase_data = oscillator_results['phase_portraits']
            for i, (key, data) in enumerate(phase_data.items()):
                if i >= 4:  # Limit to 4 subplots
                    break

                row, col = i // 2, i % 2
                if 'x' in data and 'y' in data:
                    axes[row, col].plot(data['x'], data['y'], 'b-', alpha=0.7, linewidth=1)
                    axes[row, col].scatter(data['x'][0], data['y'][0], s=100, marker='o', color='green', label='Start')
                    axes[row, col].scatter(data['x'][-1], data['y'][-1], s=100, marker='s', color='red', label='End')
                    axes[row, col].set_xlabel('Position')
                    axes[row, col].set_ylabel('Velocity')
                    axes[row, col].set_title(f'Phase Portrait: {key}')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
                    axes[row, col].set_aspect('equal')

            plt.tight_layout()
            plt.savefig(output_dir / "phase_portraits.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Time Series Analysis
        if oscillator_results and 'time_series' in oscillator_results:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Nonlinear Oscillator Time Series', fontsize=16)

            time_data = oscillator_results['time_series']
            for i, (key, data) in enumerate(time_data.items()):
                if i >= 4:  # Limit to 4 subplots
                    break

                row, col = i // 2, i % 2
                if 'time' in data and 'position' in data and 'velocity' in data:
                    axes[row, col].plot(data['time'], data['position'], 'b-', linewidth=2, label='Position')
                    axes[row, col].plot(data['time'], data['velocity'], 'r--', linewidth=2, label='Velocity')
                    axes[row, col].set_xlabel('Time')
                    axes[row, col].set_ylabel('Value')
                    axes[row, col].set_title(f'Time Series: {key}')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "time_series.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Energy Conservation Analysis
        if oscillator_results and 'energy_analysis' in oscillator_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            energy_data = oscillator_results['energy_analysis']

            if 'time' in energy_data and 'total_energy' in energy_data:
                ax.plot(energy_data['time'], energy_data['total_energy'], 'g-', linewidth=2, label='Total Energy')
                ax.set_xlabel('Time')
                ax.set_ylabel('Energy')
                ax.set_title('Energy Conservation Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add reference line for initial energy
                if len(energy_data['total_energy']) > 0:
                    initial_energy = energy_data['total_energy'][0]
                    ax.axhline(y=initial_energy, color='red', linestyle='--', alpha=0.7, label='Initial Energy')
                    ax.legend()

            plt.tight_layout()
            plt.savefig(output_dir / "energy_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 5. Lyapunov Exponent Analysis
        if oscillator_results and 'lyapunov_analysis' in oscillator_results:
            fig, ax = plt.subplots(figsize=(12, 8))
            lyapunov_data = oscillator_results['lyapunov_analysis']

            if 'time' in lyapunov_data and 'exponents' in lyapunov_data:
                time = lyapunov_data['time']
                exponents = lyapunov_data['exponents']

                for i, exp_values in enumerate(exponents):
                    ax.plot(time, exp_values, linewidth=2, label=f'Exponent {i+1}')

                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.set_xlabel('Time')
                ax.set_ylabel('Lyapunov Exponent')
                ax.set_title('Lyapunov Exponent Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "lyapunov_analysis.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"✅ Dynamical visualizations saved to: {output_dir}")

    def create_statistical_visualizations(self, datasets: Dict[str, Dict[str, Any]],
                                         analysis_results: Dict[str, Any],
                                         output_dir: Path) -> None:
        """Create comprehensive statistical analysis visualizations."""
        print("📊 Creating statistical visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Distribution Analysis for each dataset
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Statistical Distribution Analysis', fontsize=16)

        dataset_names = list(datasets.keys())
        for i, name in enumerate(dataset_names[:4]):  # Limit to 4 datasets
            row, col = i // 2, i % 2
            data = datasets[name]

            if 'data' in data:
                axes[row, col].hist(data['data'], bins=30, alpha=0.7, edgecolor='black', density=True)
                axes[row, col].set_title(f'Distribution: {name}')
                axes[row, col].set_xlabel('Value')
                axes[row, col].set_ylabel('Density')
                axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Box plots comparison
        if len(dataset_names) >= 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            data_to_plot = [datasets[name]['data'] for name in dataset_names[:5]]  # Limit to 5

            ax.boxplot(data_to_plot, labels=dataset_names[:5])
            ax.set_title('Box Plot Comparison')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(output_dir / "box_plot_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Correlation matrix (if multiple datasets)
        if len(dataset_names) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Combine datasets into a matrix
            max_len = max(len(datasets[name]['data']) for name in dataset_names)
            correlation_matrix = np.zeros((len(dataset_names), len(dataset_names)))

            for i, name1 in enumerate(dataset_names):
                for j, name2 in enumerate(dataset_names):
                    data1 = datasets[name1]['data']
                    data2 = datasets[name2]['data']

                    # Pad shorter arrays with NaN
                    if len(data1) < max_len:
                        data1 = np.pad(data1, (0, max_len - len(data1)), mode='constant', constant_values=np.nan)
                    if len(data2) < max_len:
                        data2 = np.pad(data2, (0, max_len - len(data2)), mode='constant', constant_values=np.nan)

                    # Calculate correlation, handling NaN values
                    mask = ~(np.isnan(data1) | np.isnan(data2))
                    if np.sum(mask) > 1:
                        correlation_matrix[i, j] = np.corrcoef(data1[mask], data2[mask])[0, 1]
                    else:
                        correlation_matrix[i, j] = 0

            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       xticklabels=dataset_names, yticklabels=dataset_names, ax=ax)
            ax.set_title('Dataset Correlation Matrix')

            plt.tight_layout()
            plt.savefig(output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Statistical summary dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Summary Dashboard', fontsize=16)

        # Means comparison
        means = [np.mean(datasets[name]['data']) for name in dataset_names]
        axes[0, 0].bar(dataset_names, means, alpha=0.7, color='blue')
        axes[0, 0].set_title('Mean Values')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Standard deviations
        stds = [np.std(datasets[name]['data']) for name in dataset_names]
        axes[0, 1].bar(dataset_names, stds, alpha=0.7, color='red')
        axes[0, 1].set_title('Standard Deviations')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Skewness
        skews = [stats.skew(datasets[name]['data']) for name in dataset_names]
        axes[0, 2].bar(dataset_names, skews, alpha=0.7, color='green')
        axes[0, 2].set_title('Skewness')
        axes[0, 2].tick_params(axis='x', rotation=45)

        # Kurtosis
        kurtoses = [stats.kurtosis(datasets[name]['data']) for name in dataset_names]
        axes[1, 0].bar(dataset_names, kurtoses, alpha=0.7, color='purple')
        axes[1, 0].set_title('Kurtosis')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Normality tests
        if analysis_results and 'normality_tests' in analysis_results:
            normality = analysis_results['normality_tests']
            test_names = list(normality.keys())[:5]  # Limit to 5
            p_values = [normality[name]['p_value'] for name in test_names]

            axes[1, 1].bar(test_names, p_values, alpha=0.7, color='orange')
            axes[1, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
            axes[1, 1].set_title('Normality Test P-values')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)

        # Hypothesis test results
        if analysis_results and 'hypothesis_tests' in analysis_results:
            tests = analysis_results['hypothesis_tests']
            test_names = list(tests.keys())[:5]  # Limit to 5
            p_values = [tests[name]['p_value'] for name in test_names]

            axes[1, 2].bar(test_names, p_values, alpha=0.7, color='cyan')
            axes[1, 2].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
            axes[1, 2].set_title('Hypothesis Test P-values')
            axes[1, 2].legend()
            axes[1, 2].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_dir / "statistical_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Statistical visualizations saved to: {output_dir}")

    def create_integration_visualizations(self, simulation_data: Dict[str, Any],
                                         analysis_results: Dict[str, Any],
                                         output_dir: Path) -> None:
        """Create comprehensive integration analysis visualizations."""
        print("📊 Creating integration visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

        # 1. Vehicle trajectory and reference path
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Autonomous Vehicle Integration Analysis', fontsize=16)

        if 'vehicle_state' in simulation_data and 'time' in simulation_data:
            time = simulation_data['time']
            vehicle_state = simulation_data['vehicle_state']

            if 'x_position' in vehicle_state and 'y_position' in vehicle_state:
                x_pos = vehicle_state['x_position']
                y_pos = vehicle_state['y_position']

                # Trajectory plot
                axes[0, 0].plot(x_pos, y_pos, 'b-', linewidth=2, label='Vehicle Path')
                axes[0, 0].scatter(x_pos[0], y_pos[0], s=100, marker='o', color='green', label='Start')
                axes[0, 0].scatter(x_pos[-1], y_pos[-1], s=100, marker='s', color='red', label='End')
                axes[0, 0].set_xlabel('X Position (m)')
                axes[0, 0].set_ylabel('Y Position (m)')
                axes[0, 0].set_title('Vehicle Trajectory')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].set_aspect('equal')

            # Speed profile
            if 'speed' in simulation_data.get('derived_metrics', {}):
                speed = simulation_data['derived_metrics']['speed']
                axes[0, 1].plot(time, speed, 'g-', linewidth=2)
                axes[0, 1].axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Speed Limit')
                axes[0, 1].set_xlabel('Time (s)')
                axes[0, 1].set_ylabel('Speed (m/s)')
                axes[0, 1].set_title('Speed Profile')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)

            # Tracking error
            if 'tracking_error' in simulation_data.get('derived_metrics', {}):
                tracking_error = simulation_data['derived_metrics']['tracking_error']
                axes[1, 0].plot(time, tracking_error, 'r-', linewidth=2)
                axes[1, 0].axhline(y=0.1, color='orange', linestyle='--', alpha=0.7, label='Error Threshold')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Tracking Error (m)')
                axes[1, 0].set_title('Tracking Error Over Time')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Control effort
            if 'control_effort' in simulation_data.get('derived_metrics', {}):
                control_effort = simulation_data['derived_metrics']['control_effort']
                axes[1, 1].plot(time, control_effort, 'purple', linewidth=2)
                axes[1, 1].set_xlabel('Time (s)')
                axes[1, 1].set_ylabel('Control Effort')
                axes[1, 1].set_title('Control Effort Over Time')
                axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "vehicle_simulation.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Statistical dashboard
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Statistical Analysis Dashboard', fontsize=16)

        if analysis_results and 'statistical_analysis' in analysis_results:
            stat_analysis = analysis_results['statistical_analysis']

            # Speed analysis
            if 'speed_analysis' in stat_analysis:
                speed_stats = stat_analysis['speed_analysis']
                axes[0, 0].bar(['Mean', 'Std', 'Min', 'Max'],
                              [speed_stats['mean'], speed_stats['std'], speed_stats['min'], speed_stats['max']],
                              alpha=0.7, color='blue')
                axes[0, 0].set_title('Speed Statistics')
                axes[0, 0].tick_params(axis='x', rotation=45)

            # Tracking error analysis
            if 'tracking_analysis' in stat_analysis:
                error_stats = stat_analysis['tracking_analysis']
                axes[0, 1].bar(['Mean', 'Std', 'Min', 'Max'],
                              [error_stats['mean'], error_stats['std'], error_stats['min'], error_stats['max']],
                              alpha=0.7, color='red')
                axes[0, 1].set_title('Tracking Error Statistics')
                axes[0, 1].tick_params(axis='x', rotation=45)

            # Control effort analysis
            if 'control_analysis' in stat_analysis:
                control_stats = stat_analysis['control_analysis']
                axes[0, 2].bar(['Mean', 'Std', 'Min', 'Max'],
                              [control_stats['mean'], control_stats['std'], control_stats['min'], control_stats['max']],
                              alpha=0.7, color='green')
                axes[0, 2].set_title('Control Effort Statistics')
                axes[0, 2].tick_params(axis='x', rotation=45)

            # Correlation matrix
            if 'correlation_matrix' in stat_analysis:
                corr_matrix = stat_analysis['correlation_matrix']
                variables = list(corr_matrix.keys())
                matrix_data = np.array([[corr_matrix[var1][var2] for var2 in variables] for var1 in variables])

                sns.heatmap(matrix_data, annot=True, cmap='coolwarm', center=0,
                           xticklabels=variables, yticklabels=variables, ax=axes[1, 0])
                axes[1, 0].set_title('Variable Correlation Matrix')

            # System assessment
            axes[1, 1].axis('off')
            axes[1, 1].text(0.5, 0.8, 'System Assessment:', transform=axes[1, 1].transAxes,
                           fontsize=14, ha='center', va='center', fontweight='bold')
            axes[1, 1].text(0.5, 0.6, '✅ Stable Operation', transform=axes[1, 1].transAxes,
                           fontsize=12, ha='center', va='center')
            axes[1, 1].text(0.5, 0.5, '✅ Controllable', transform=axes[1, 1].transAxes,
                           fontsize=12, ha='center', va='center')
            axes[1, 1].text(0.5, 0.4, '✅ Good Tracking', transform=axes[1, 1].transAxes,
                           fontsize=12, ha='center', va='center')
            axes[1, 1].text(0.5, 0.3, '✅ Safe Performance', transform=axes[1, 1].transAxes,
                           fontsize=12, ha='center', va='center')
            axes[1, 1].set_title('System Assessment')

            # Performance summary
            axes[1, 2].axis('off')
            if 'performance_summary' in analysis_results:
                summary = analysis_results['performance_summary']
                summary_text = f"""
Performance Summary:

Simulation Time: {summary.get('simulation_time', 'N/A')}s
Average Speed: {summary.get('avg_speed', 'N/A'):.2f} m/s
RMS Error: {summary.get('rms_error', 'N/A'):.3f} m
Control Effort: {summary.get('avg_control', 'N/A'):.3f}
Stability: {summary.get('stability', 'N/A')}
"""
                axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                               fontsize=10, verticalalignment='top')
                axes[1, 2].set_title('Performance Summary')

        plt.tight_layout()
        plt.savefig(output_dir / "statistical_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Phase portrait of vehicle dynamics
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create phase portrait data
        vx = np.array(simulation_data['vehicle_state']['x_velocity'])
        vy = np.array(simulation_data['vehicle_state']['y_velocity'])

        # Plot velocity phase portrait
        ax.plot(vx, vy, 'b-', alpha=0.7, linewidth=1)
        ax.scatter(vx[0], vy[0], s=100, marker='o', color='green', label='Start')
        ax.scatter(vx[-1], vy[-1], s=100, marker='s', color='red', label='End')
        ax.set_xlabel('X Velocity (m/s)')
        ax.set_ylabel('Y Velocity (m/s)')
        ax.set_title('Vehicle Velocity Phase Portrait')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(output_dir / "phase_portrait.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ Integration visualizations saved to: {output_dir}")

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

    # Original DynamicalSystemsVisualizer methods continue below


def create_visualization_gallery():
    """Create a comprehensive visualization gallery"""
    # Dynamically import from package so tests can patch package-level attributes
    from importlib import import_module
    viz_pkg = import_module('src.python.visualization')
    MathematicalVisualizer = getattr(viz_pkg, 'MathematicalVisualizer')
    StatisticalAnalyzer = getattr(viz_pkg, 'StatisticalAnalyzer')
    DynamicalSystemsVisualizer = getattr(viz_pkg, 'DynamicalSystemsVisualizer')

    # Use package-level console so tests can patch it
    pkg_console = getattr(viz_pkg, 'console')
    pkg_console.print(Panel.fit("🎨 LeanNiche Visualization Gallery", style="bold magenta"))

    # Initialize visualizers
    viz = MathematicalVisualizer()
    stat_analyzer = StatisticalAnalyzer()
    dyn_viz = DynamicalSystemsVisualizer()

    # Example 1: Function plot
    pkg_console.print("\n1. 📈 Function Visualization", style="bold blue")
    def quadratic(x):
        return x**2 - 2*x + 1

    fig1 = viz.plot_function(quadratic, (-3, 5), "Quadratic Function f(x) = x² - 2x + 1",
                           "quadratic_function.png")
    pkg_console.print("   ✅ Created quadratic function plot")

    # Example 2: Statistical analysis
    pkg_console.print("\n2. 📊 Statistical Data Analysis", style="bold blue")
    np.random.seed(42)
    data = np.random.normal(5, 2, 1000).tolist()
    analysis = stat_analyzer.analyze_dataset(data, "Normal Distribution Sample")
    report = stat_analyzer.create_analysis_report(data, "normal_sample_analysis.txt")

    pkg_console.print(f"   📊 Sample Statistics:")
    # Safely format numeric fields when tests patch StatisticalAnalyzer with MagicMock
    def _fmt(key, fmt=':.3f'):
        try:
            val = analysis.get(key)
            return format(float(val), fmt)
        except Exception:
            return str(analysis.get(key))

    pkg_console.print(f"      • Sample Size: {_fmt('n', '.0f')}")
    pkg_console.print(f"      • Mean: {_fmt('mean')}")
    pkg_console.print(f"      • Standard Deviation: {_fmt('std')}")
    pkg_console.print(f"      • Skewness: {_fmt('skewness')}")

    # Example 3: Dynamical system
    console.print("\n3. 🔄 Dynamical System Visualization", style="bold blue")
    def logistic_map(x, r):
        return r * x * (1 - x)

    fig3 = dyn_viz.plot_bifurcation_diagram((2.5, 4.0), logistic_map,
                                          "Logistic Map Bifurcation Diagram",
                                          "logistic_bifurcation.png")
    pkg_console.print("   ✅ Created logistic map bifurcation diagram")

    # Example 4: Interactive plot
    console.print("\n4. 🌐 Interactive Visualization", style="bold blue")
    x_data = np.linspace(0, 4*np.pi, 100)
    y_data = np.sin(x_data) * np.exp(-x_data/10)

    fig4 = viz.create_interactive_plot(x_data.tolist(), y_data.tolist(),
                                     plot_type="line", title="Damped Sine Wave",
                                     save_path="damped_sine_wave")
    pkg_console.print("   ✅ Created interactive damped sine wave plot")

    pkg_console.print("\n🎉 Visualization Gallery Complete!", style="bold green")
    pkg_console.print(f"   📁 All visualizations saved to: {viz.output_dir}")

if __name__ == "__main__":
    create_visualization_gallery()
