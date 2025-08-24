"""Tests for LeanNiche visualization modules"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from src.python.visualization import (
    MathematicalVisualizer,
    StatisticalAnalyzer,
    DynamicalSystemsVisualizer,
    create_visualization_gallery
)


class TestMathematicalVisualizer:
    """Test MathematicalVisualizer class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.viz = MathematicalVisualizer()

    def test_initialization(self):
        """Test visualizer initialization"""
        assert self.viz.output_dir.exists()
        assert isinstance(self.viz.output_dir, type(self.viz.output_dir))

    def test_plot_function(self):
        """Test function plotting"""
        def f(x):
            return x**2

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = self.viz.plot_function(f, (-5, 5), "Test Function", "test.png")

            mock_subplots.assert_called_once()
            mock_ax.plot.assert_called_once()
            mock_ax.set_title.assert_called_once_with("Test Function")
            mock_savefig.assert_called_once()
            assert result == mock_fig

    def test_plot_statistical_data(self):
        """Test statistical data plotting"""
        data = [1, 2, 3, 4, 5]

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.savefig') as mock_savefig, \
             patch('matplotlib.pyplot.tight_layout') as mock_layout:

            mock_fig = MagicMock()
            mock_axes = [MagicMock(), MagicMock()]
            mock_subplots.return_value = (mock_fig, mock_axes)

            result = self.viz.plot_statistical_data(data, "Test Data", "test_stats.png")

            mock_subplots.assert_called_once()
            assert result == mock_fig

    def test_create_interactive_plot(self):
        """Test interactive plot creation"""
        x_data = [1, 2, 3, 4, 5]
        y_data = [1, 4, 9, 16, 25]

        with patch('plotly.graph_objects.Figure') as mock_fig:
            mock_figure = MagicMock()
            mock_fig.return_value = mock_figure

            result = self.viz.create_interactive_plot(
                x_data, y_data, "scatter", "Test Interactive", "test_interactive"
            )

            mock_fig.assert_called_once()
            mock_figure.update_layout.assert_called_once()
            assert result == mock_figure

    def test_visualize_network(self):
        """Test network visualization"""
        adjacency_matrix = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('networkx.from_numpy_array') as mock_nx, \
             patch('networkx.spring_layout') as mock_layout, \
             patch('networkx.draw') as mock_draw:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = self.viz.visualize_network(
                adjacency_matrix, "Test Network", "test_network.png"
            )

            mock_nx.assert_called_once_with(adjacency_matrix)
            mock_draw.assert_called_once()
            assert result == mock_fig


class TestStatisticalAnalyzer:
    """Test StatisticalAnalyzer class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = StatisticalAnalyzer()
        self.test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    def test_analyze_dataset(self):
        """Test dataset analysis"""
        result = self.analyzer.analyze_dataset(self.test_data, "Test Data")

        assert result['name'] == "Test Data"
        assert result['n'] == 10
        assert result['mean'] == 5.5
        assert result['median'] == 5.5
        assert result['min'] == 1
        assert result['max'] == 10
        assert 'std' in result
        assert 'q25' in result
        assert 'q75' in result
        assert 'iqr' in result

    def test_create_analysis_report(self):
        """Test analysis report creation"""
        with patch('builtins.open') as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            result = self.analyzer.create_analysis_report(
                self.test_data, "Test Analysis", "test_report.txt"
            )

            assert "📊 Statistical Analysis Report: Test Analysis" in result
            assert "Sample Size: 10" in result
            assert "Mean: 5.5" in result
            mock_file.write.assert_called_once()


class TestDynamicalSystemsVisualizer:
    """Test DynamicalSystemsVisualizer class"""

    def setup_method(self):
        """Setup test fixtures"""
        self.dyn_viz = DynamicalSystemsVisualizer()

    def test_plot_phase_portrait(self):
        """Test phase portrait plotting"""
        def vector_field_x(x, y):
            return -y

        def vector_field_y(x, y):
            return x

        domain = ((-2, 2), (-2, 2))

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('numpy.linspace') as mock_linspace, \
             patch('numpy.meshgrid') as mock_meshgrid:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_linspace.return_value = np.array([-2, -1, 0, 1, 2])
            mock_meshgrid.return_value = (
                np.array([[1, 2], [3, 4]]),
                np.array([[5, 6], [7, 8]])
            )

            result = self.dyn_viz.plot_phase_portrait(
                vector_field_x, vector_field_y, domain, "Test Phase Portrait"
            )

            mock_subplots.assert_called_once()
            mock_ax.quiver.assert_called_once()
            assert result == mock_fig

    def test_plot_bifurcation_diagram(self):
        """Test bifurcation diagram plotting"""
        def logistic_map(x, r):
            return r * x * (1 - x)

        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('numpy.linspace') as mock_linspace:

            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            result = self.dyn_viz.plot_bifurcation_diagram(
                (2.5, 4.0), logistic_map, "Test Bifurcation"
            )

            mock_subplots.assert_called_once()
            mock_ax.set_title.assert_called_once_with("Test Bifurcation")
            assert result == mock_fig


class TestGallery:
    """Test gallery creation"""

    def test_create_visualization_gallery(self):
        """Test gallery creation function"""
        with patch('src.python.visualization.MathematicalVisualizer') as mock_viz, \
             patch('src.python.visualization.StatisticalAnalyzer') as mock_stat, \
             patch('src.python.visualization.DynamicalSystemsVisualizer') as mock_dyn:

            # Mock the classes
            mock_viz_instance = MagicMock()
            mock_stat_instance = MagicMock()
            mock_dyn_instance = MagicMock()

            mock_viz.return_value = mock_viz_instance
            mock_stat.return_value = mock_stat_instance
            mock_dyn.return_value = mock_dyn_instance

            # Mock console
            with patch('src.python.visualization.console') as mock_console:
                create_visualization_gallery()

                # Check that visualizers were created
                mock_viz.assert_called_once()
                mock_stat.assert_called_once()
                mock_dyn.assert_called_once()

                # Check that print statements were called
                mock_console.print.assert_called()
