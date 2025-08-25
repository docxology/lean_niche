import click
from pathlib import Path
import json

from src.python.visualization import (
    MathematicalVisualizer,
    StatisticalAnalyzer,
    DynamicalSystemsVisualizer,
    create_visualization_gallery,
    console as viz_console,
)

# Re-export for tests
console = viz_console


@click.group(help='LeanNiche Mathematical Visualization')
def cli():
    """LeanNiche CLI"""
    pass


@cli.command(name='plot-function')
@click.option('--function', required=True)
@click.option('--domain', default='-5,5')
@click.option('--title', default='Function Plot')
def plot_function(function, domain, title):
    # Use real visualizer
    try:
        viz = MathematicalVisualizer()
        parts = domain.split(',')
        domain_tuple = (float(parts[0]), float(parts[1])) if len(parts) == 2 else (-5.0, 5.0)
        # build callable from expression
        f = eval(function, {"__builtins__": {}}, {})
        viz.plot_function(f, domain_tuple, title)
        print('✅ Function plot created')
    except Exception as e:
        print(f"❌ Error creating function plot: {e}")
        raise


@cli.command(name='analyze-data')
@click.option('--data', default=None)
@click.option('--output', default=None)
def analyze_data(data, output):
    # Use real analyzer
    if not data:
        print('Please provide data file with --data option')
        raise SystemExit(1)

    try:
        with open(data, 'r') as f:
            d = json.load(f)
        analyzer = StatisticalAnalyzer()
        analysis = analyzer.analyze_dataset(d)
        if output:
            analyzer.create_analysis_report(d, output)
        print('📊 Statistical Analysis Results:')
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")
        raise


@cli.command()
def gallery():
    create_visualization_gallery()
    print('🎨 Visualization gallery created')


@cli.command()
def info():
    print('🔍 LeanNiche System Information:')


def main():
    cli()


if __name__ == '__main__':
    main()


