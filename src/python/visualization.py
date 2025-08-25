#!/usr/bin/env python3
"""
LeanNiche Visualization Module
Provides visualization utilities for mathematical data and Lean proofs.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


class LeanNicheVisualizer:
    """Visualization utilities for LeanNiche data"""

    def __init__(self):
        """Initialize the visualizer"""
        self.style = 'default'
        plt.style.use(self.style)

    def plot_time_series(self, time: List[float], values: List[float],
                        title: str = "Time Series", xlabel: str = "Time",
                        ylabel: str = "Value", save_path: Optional[str] = None):
        """Plot a time series"""
        plt.figure(figsize=(10, 6))
        plt.plot(time, values, 'b-', linewidth=2)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()

    def plot_comparison(self, data: Dict[str, List[float]],
                       title: str = "Comparison", xlabel: str = "X",
                       ylabel: str = "Y", save_path: Optional[str] = None):
        """Plot multiple data series for comparison"""
        plt.figure(figsize=(12, 8))

        for label, values in data.items():
            plt.plot(values, label=label, linewidth=2)

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        else:
            plt.show()

    def plot_histogram(self, data: List[float], bins: int = 50,
                      title: str = "Histogram", xlabel: str = "Value",
                      ylabel: str = "Frequency", save_path: Optional[str] = None):
        """Plot a histogram"""
        plt.figure(figsize=(10, 6))
        plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Histogram saved to: {save_path}")
        else:
            plt.show()

    def create_dashboard(self, plots_data: Dict[str, Any], save_path: str):
        """Create a comprehensive dashboard with multiple plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("LeanNiche Analysis Dashboard", fontsize=16)

        # Plot 1: Time series if available
        if 'time_series' in plots_data:
            ax = axes[0, 0]
            ts_data = plots_data['time_series']
            ax.plot(ts_data.get('time', []), ts_data.get('values', []))
            ax.set_title("Time Series")
            ax.grid(True, alpha=0.3)

        # Plot 2: Comparison if available
        if 'comparison' in plots_data:
            ax = axes[0, 1]
            comp_data = plots_data['comparison']
            for label, values in comp_data.items():
                ax.plot(values, label=label)
            ax.set_title("Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Distribution if available
        if 'distribution' in plots_data:
            ax = axes[1, 0]
            dist_data = plots_data['distribution']
            ax.hist(dist_data, bins=30, alpha=0.7)
            ax.set_title("Distribution")
            ax.grid(True, alpha=0.3)

        # Plot 4: Statistics summary
        if 'statistics' in plots_data:
            ax = axes[1, 1]
            stats = plots_data['statistics']
            ax.axis('off')

            # Create a table with statistics
            table_data = []
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    table_data.append([key, ".3f"])

            ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                    loc='center', cellLoc='center')
            ax.set_title("Statistics Summary")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        else:
            plt.show()


def create_sample_visualization():
    """Create a sample visualization for testing"""
    visualizer = LeanNicheVisualizer()

    # Sample time series
    time = np.linspace(0, 10, 100)
    values = np.sin(time) * np.exp(-time/5)

    visualizer.plot_time_series(
        time=time,
        values=values,
        title="Sample Exponential Decay Sine Wave",
        xlabel="Time (s)",
        ylabel="Amplitude",
        save_path="sample_visualization.png"
    )

    print("Sample visualization created successfully!")


if __name__ == "__main__":
    create_sample_visualization()
