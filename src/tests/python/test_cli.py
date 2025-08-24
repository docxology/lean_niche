"""Tests for LeanNiche CLI module"""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from src.python.cli import cli, plot_function, analyze_data, gallery


class TestCLI:
    """Test CLI commands"""

    def test_cli_main(self):
        """Test CLI main entry point"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'LeanNiche Mathematical Visualization' in result.output

    def test_plot_function_command(self):
        """Test plot function command"""
        runner = CliRunner()
        with patch('src.python.cli.MathematicalVisualizer') as mock_viz:
            mock_viz.return_value.plot_function.return_value = MagicMock()
            result = runner.invoke(plot_function, [
                '--function', 'lambda x: x**2',
                '--domain', '-5,5',
                '--title', 'Test Plot'
            ])
            assert result.exit_code == 0
            assert '✅ Function plot created' in result.output

    def test_analyze_data_command_no_file(self):
        """Test analyze data command without file"""
        runner = CliRunner()
        result = runner.invoke(analyze_data)
        assert result.exit_code != 0
        assert 'Please provide data file' in result.output

    def test_analyze_data_command_with_file(self):
        """Test analyze data command with file"""
        runner = CliRunner()
        with patch('builtins.open'), \
             patch('json.load') as mock_load, \
             patch('src.python.cli.StatisticalAnalyzer') as mock_analyzer:

            mock_load.return_value = [1, 2, 3, 4, 5]
            mock_analyzer.return_value.analyze_dataset.return_value = {
                'n': 5, 'mean': 3.0, 'std': 1.41, 'min': 1, 'max': 5
            }

            result = runner.invoke(analyze_data, ['--data', 'test_data.json'])
            assert result.exit_code == 0
            assert '📊 Statistical Analysis Results:' in result.output

    def test_gallery_command(self):
        """Test gallery command"""
        runner = CliRunner()
        with patch('src.python.cli.create_visualization_gallery') as mock_gallery:
            mock_gallery.return_value = None
            result = runner.invoke(gallery)
            assert result.exit_code == 0
            assert '🎨 Visualization gallery created' in result.output

    def test_info_command(self):
        """Test info command"""
        runner = CliRunner()
        result = runner.invoke(cli, ['info'])
        assert result.exit_code == 0
        assert '🔍 LeanNiche System Information:' in result.output
