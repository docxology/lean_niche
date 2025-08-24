"""Command Line Interface for LeanNiche

CLI tools for visualization, analysis, and utilities.
"""

import click
import json
import os
from pathlib import Path
from typing import Optional
from ..visualization.visualization import (
    MathematicalVisualizer,
    StatisticalAnalyzer,
    DynamicalSystemsVisualizer,
    create_visualization_gallery,
)

@click.group()
@click.option('--output-dir', default='visualizations', help='Output directory for visualizations')
@click.pass_context
def cli(ctx, output_dir):
    """LeanNiche Mathematical Visualization and Analysis Tools"""
    ctx.ensure_object(dict)
    ctx.obj['output_dir'] = output_dir

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)

@cli.command()
@click.option('--function', default='lambda x: x**2', help='Function to plot (Python expression)')
@click.option('--domain', default='-5,5', help='Domain as "min,max"')
@click.option('--title', default='Function Plot', help='Plot title')
@click.option('--output', help='Output filename')
@click.pass_context
def plot_function(ctx, function, domain, title, output):
    """Plot a mathematical function"""
    try:
        # Parse domain
        domain_parts = domain.split(',')
        domain_min, domain_max = float(domain_parts[0]), float(domain_parts[1])

        # Create function from string
        def f(x):
            return eval(function, {"__builtins__": {}}, {"x": x})

        # Create visualizer and plot
        viz = MathematicalVisualizer()
        fig = viz.plot_function(f, (domain_min, domain_max), title, output)

        click.echo(f"✅ Function plot created: {title}")
        if output:
            click.echo(f"📁 Saved to: {viz.output_dir / output}")

    except Exception as e:
        click.echo(f"❌ Error plotting function: {e}", err=True)

@cli.command()
@click.option('--data', help='JSON file containing data array')
@click.option('--output', help='Output filename for analysis')
@click.pass_context
def analyze_data(ctx, data, output):
    """Perform statistical analysis on data"""
    try:
        # Load data
        if data:
            with open(data, 'r') as f:
                data_list = json.load(f)
        else:
            click.echo("Please provide data file with --data option", err=True)
            return

        # Perform analysis
        analyzer = StatisticalAnalyzer()
        analysis = analyzer.analyze_dataset(data_list)

        # Display results
        click.echo("📊 Statistical Analysis Results:")
        click.echo(f"   Sample Size: {analysis['n']}")
        click.echo(f"   Mean: {analysis['mean']:.4f}")
        click.echo(f"   Standard Deviation: {analysis['std']:.4f}")
        click.echo(f"   Min: {analysis['min']:.4f}")
        click.echo(f"   Max: {analysis['max']:.4f}")

        if output:
            report = analyzer.create_analysis_report(data_list, output)
            click.echo(f"📄 Detailed report saved to: {output}")

    except Exception as e:
        click.echo(f"❌ Error analyzing data: {e}", err=True)

@cli.command()
@click.option('--matrix-file', help='JSON file containing adjacency matrix')
@click.option('--output', help='Output filename')
@click.pass_context
def visualize_network(ctx, matrix_file, output):
    """Visualize a network/graph"""
    try:
        if not matrix_file:
            click.echo("Please provide matrix file with --matrix-file option", err=True)
            return

        # Load matrix
        with open(matrix_file, 'r') as f:
            matrix = json.load(f)

        import numpy as np
        matrix = np.array(matrix)

        # Create visualization
        viz = MathematicalVisualizer()
        fig = viz.visualize_network(matrix, "Network Visualization", output)

        click.echo("✅ Network visualization created")
        if output:
            click.echo(f"📁 Saved to: {viz.output_dir / output}")

    except Exception as e:
        click.echo(f"❌ Error visualizing network: {e}", err=True)

@cli.command()
def gallery():
    """Create a comprehensive visualization gallery"""
    try:
        create_visualization_gallery()
        click.echo("🎨 Visualization gallery created successfully!")
    except Exception as e:
        click.echo(f"❌ Error creating gallery: {e}", err=True)

@cli.command()
@click.option('--format', type=click.Choice(['json', 'csv', 'txt']),
              default='json', help='Export format')
@click.option('--output', help='Output filename')
@click.pass_context
def export_examples(ctx, format, output):
    """Export example datasets for testing"""
    try:
        import numpy as np

        # Generate example datasets
        datasets = {
            'normal': np.random.normal(5, 2, 1000).tolist(),
            'uniform': np.random.uniform(0, 10, 1000).tolist(),
            'exponential': np.random.exponential(2, 1000).tolist(),
            'bimodal': (np.random.normal(3, 1, 500).tolist() +
                       np.random.normal(7, 1, 500).tolist())
        }

        if format == 'json':
            data = json.dumps(datasets, indent=2)
            ext = 'json'
        elif format == 'csv':
            import pandas as pd
            df = pd.DataFrame(datasets)
            data = df.to_csv(index=False)
            ext = 'csv'
        else:  # txt
            data = "\n".join([f"{k}: {v[:10]}..." for k, v in datasets.items()])
            ext = 'txt'

        output_file = output or f"example_datasets.{ext}"
        with open(output_file, 'w') as f:
            f.write(data)

        click.echo(f"✅ Example datasets exported to: {output_file}")

    except Exception as e:
        click.echo(f"❌ Error exporting examples: {e}", err=True)

@cli.command()
def info():
    """Display system information"""
    import sys
    import matplotlib
    import numpy
    import seaborn
    import plotly

    click.echo("🔍 LeanNiche System Information:")
    click.echo(f"   Python Version: {sys.version}")
    click.echo(f"   Matplotlib: {matplotlib.__version__}")
    click.echo(f"   NumPy: {numpy.__version__}")
    click.echo(f"   Seaborn: {seaborn.__version__}")
    click.echo(f"   Plotly: {plotly.__version__}")
    click.echo(f"   Working Directory: {os.getcwd()}")

def main():
    """Main entry point for the CLI"""
    cli()

if __name__ == "__main__":
    main()
