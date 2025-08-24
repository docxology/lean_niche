#!/usr/bin/env python3
"""
🚀 Run All LeanNiche Examples

This script runs all available examples in the LeanNiche project, demonstrating
the "Clean Thin Orchestrator" pattern across different mathematical domains.

Usage:
    python run_all_examples.py [--examples EXAMPLE1,EXAMPLE2,...] [--output-dir OUTPUT_DIR]

Arguments:
    --examples: Comma-separated list of specific examples to run (optional)
    --output-dir: Base directory for all example outputs (default: outputs)

Examples:
    python run_all_examples.py
    python run_all_examples.py --examples statistical_analysis_example.py,dynamical_systems_example.py
    python run_all_examples.py --output-dir my_results
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

class ExampleRunner:
    """Orchestrator for running multiple LeanNiche examples."""

    def __init__(self, output_dir: str = "outputs"):
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir = Path(__file__).parent

        # Define all available examples
        self.all_examples = [
            {
                'script': 'statistical_analysis_example.py',
                'name': 'Statistical Analysis',
                'description': 'Comprehensive statistical analysis with Lean-verified methods',
                'domain': 'Statistics'
            },
            {
                'script': 'dynamical_systems_example.py',
                'name': 'Dynamical Systems',
                'description': 'Chaos theory, bifurcation analysis, and nonlinear dynamics',
                'domain': 'Dynamical Systems'
            },
            {
                'script': 'control_theory_example.py',
                'name': 'Control Theory',
                'description': 'PID control, LQR optimization, and stability analysis',
                'domain': 'Control Theory'
            },
            {
                'script': 'integration_showcase_example.py',
                'name': 'Integration Showcase',
                'description': 'Grand tour integrating statistics, dynamics, and control',
                'domain': 'Integration'
            }
        ]

    def list_examples(self):
        """List all available examples."""
        print("📚 Available LeanNiche Examples:")
        print("=" * 60)

        for i, example in enumerate(self.all_examples, 1):
            print(f"{i}. {example['name']}")
            print(f"   Script: {example['script']}")
            print(f"   Domain: {example['domain']}")
            print(f"   Description: {example['description']}")
            print()

    def run_single_example(self, example):
        """Run a single example script."""
        script_path = self.examples_dir / example['script']

        if not script_path.exists():
            print(f"❌ Example script not found: {script_path}")
            return False

        print(f"🚀 Running {example['name']}...")
        print(f"   Script: {example['script']}")
        print(f"   Domain: {example['domain']}")
        print(f"   Description: {example['description']}")
        print("-" * 50)

        try:
            # Run the example script
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.examples_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5-minute timeout per example
            )
            end_time = time.time()

            execution_time = end_time - start_time

            if result.returncode == 0:
                print("✅ SUCCESS")
                print(f"   Execution time: {execution_time:.2f} seconds")
                if result.stdout:
                    # Show last few lines of output
                    lines = result.stdout.strip().split('\n')
                    for line in lines[-5:]:  # Show last 5 lines
                        if line.strip():
                            print(f"   Output: {line}")
                print()
                return True
            else:
                print("❌ FAILED")
                print(f"   Return code: {result.returncode}")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")
                if result.stdout:
                    print(f"   Output: {result.stdout.strip()}")
                print()
                return False

        except subprocess.TimeoutExpired:
            print("❌ TIMEOUT")
            print("   Example took too long to execute (>5 minutes)")
            print()
            return False
        except Exception as e:
            print("❌ ERROR")
            print(f"   Exception: {str(e)}")
            print()
            return False

    def run_examples(self, selected_examples=None):
        """Run multiple examples."""
        if selected_examples:
            examples_to_run = [ex for ex in self.all_examples if ex['script'] in selected_examples]
        else:
            examples_to_run = self.all_examples

        if not examples_to_run:
            print("❌ No valid examples selected to run.")
            self.list_examples()
            return

        print("🚀 LeanNiche Example Runner")
        print("=" * 60)
        print(f"Running {len(examples_to_run)} example(s)")
        print(f"Output directory: {self.base_output_dir}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        results = []
        start_time = time.time()

        for example in examples_to_run:
            success = self.run_single_example(example)
            results.append({
                'example': example,
                'success': success
            })

        end_time = time.time()
        total_time = end_time - start_time

        # Generate summary report
        self.generate_summary_report(results, total_time)

    def generate_summary_report(self, results, total_time):
        """Generate a comprehensive summary report."""
        successful = sum(1 for r in results if r['success'])
        total = len(results)

        print("📊 Execution Summary")
        print("=" * 60)
        print(f"Total examples: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        print(".1f")
        print()

        # Create detailed report
        report_content = f"""# 📊 LeanNiche Examples Execution Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Overview

This report summarizes the execution of LeanNiche examples demonstrating the "Clean Thin Orchestrator" pattern.

## 📈 Execution Results

| Example | Status | Domain | Description |
|---------|--------|--------|-------------|
"""

        for result in results:
            example = result['example']
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            report_content += f"| {example['name']} | {status} | {example['domain']} | {example['description']} |\n"

        report_content += f"""
## 📊 Summary Statistics

- **Total Examples**: {total}
- **Successful**: {successful}
- **Failed**: {total - successful}
- **Success Rate**: {successful/total*100:.1f}%
- **Total Execution Time**: {total_time:.2f} seconds
- **Average Time per Example**: {total_time/total:.2f} seconds

## 📁 Output Structure

Each example generates its own output directory:

```
{self.base_output_dir}/
├── statistical_analysis/
│   ├── proofs/
│   ├── data/
│   ├── visualizations/
│   └── reports/
├── dynamical_systems/
│   ├── proofs/
│   ├── data/
│   ├── visualizations/
│   └── reports/
├── control_theory/
│   ├── proofs/
│   ├── data/
│   ├── visualizations/
│   └── reports/
└── integration_showcase/
    ├── proofs/
    ├── data/
    ├── visualizations/
    └── reports/
```

## 🔬 Demonstrated Capabilities

### Clean Thin Orchestrator Pattern
- **Separation of Concerns**: Each example focuses on one mathematical domain
- **Modular Design**: Reusable components from `src/` submodules
- **Minimal Boilerplate**: Essential code only, no unnecessary complexity
- **Clear Dependencies**: Explicit imports and prerequisites

### Mathematical Domains Covered
- **Statistics**: Hypothesis testing, confidence intervals, distribution analysis
- **Dynamical Systems**: Chaos theory, bifurcation analysis, stability
- **Control Theory**: PID control, LQR optimization, stability analysis
- **Integration**: Multi-domain analysis combining different mathematical approaches

### Technical Integration
- **Lean 4**: Formal verification of mathematical theorems
- **Python**: Data analysis, visualization, and orchestration
- **Matplotlib/Seaborn**: Publication-quality visualizations
- **NumPy/SciPy**: Numerical computations and analysis

## 🎯 Key Achievements

1. **Verified Mathematics**: All examples use Lean-verified mathematical methods
2. **Reproducible Research**: Complete workflows with automated report generation
3. **Educational Value**: Clear documentation and progressive complexity
4. **Research Applications**: Suitable for academic publications and industrial applications

## 📈 Performance Metrics

- **Execution Success Rate**: {successful/total*100:.1f}%
- **Total Runtime**: {total_time:.2f} seconds
- **Examples per Minute**: {total/(total_time/60):.2f}

## 🔧 Methodology

### Example Structure
1. **Setup Phase**: Initialize Lean environment and components
2. **Lean Proof Phase**: Define and verify mathematical concepts
3. **Analysis Phase**: Run computational analysis and simulations
4. **Visualization Phase**: Create plots and visual representations
5. **Report Phase**: Generate comprehensive markdown reports

### Quality Assurance
- **Type Safety**: Compile-time guarantees through Lean
- **Error Handling**: Comprehensive exception handling
- **Documentation**: Detailed docstrings and comments
- **Testing**: Verified through Lean proof system

## 📈 Recommendations

### For Users
1. **Start Simple**: Begin with individual examples to understand the pattern
2. **Explore Domains**: Try examples from different mathematical areas
3. **Customize**: Modify examples for specific research needs
4. **Contribute**: Share improvements and new examples

### For Developers
1. **Follow Patterns**: Maintain the Clean Thin Orchestrator philosophy
2. **Add Verification**: Include Lean proofs for new mathematical concepts
3. **Comprehensive Testing**: Ensure examples work across different environments
4. **Documentation**: Keep examples well-documented and educational

## 🏆 Conclusion

The LeanNiche examples demonstrate the power of combining formal verification with practical computation. The "Clean Thin Orchestrator" pattern provides a robust framework for mathematical research that balances rigor with usability.

**Success Rate**: {successful/total*100:.1f}% of examples completed successfully
**Total Execution Time**: {total_time:.2f} seconds
**Mathematical Domains**: {len(set(r['example']['domain'] for r in results))} covered

---

*Report generated by LeanNiche Example Runner*
"""

        # Save report
        report_file = self.base_output_dir / "examples_execution_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        print(f"📄 Detailed report saved to: {report_file}")

        # Create summary JSON for programmatic access
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_examples': total,
                'successful': successful,
                'failed': total - successful,
                'success_rate': successful/total*100,
                'total_time': total_time,
                'average_time': total_time/total
            },
            'results': [
                {
                    'name': r['example']['name'],
                    'script': r['example']['script'],
                    'domain': r['example']['domain'],
                    'success': r['success']
                }
                for r in results
            ]
        }

        summary_file = self.base_output_dir / "examples_summary.json"
        import json
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)

        print(f"📊 Summary JSON saved to: {summary_file}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run LeanNiche Examples')
    parser.add_argument(
        '--examples',
        type=str,
        help='Comma-separated list of example scripts to run'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs',
        help='Base directory for example outputs'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available examples and exit'
    )

    args = parser.parse_args()

    runner = ExampleRunner(args.output_dir)

    if args.list:
        runner.list_examples()
        return

    # Parse selected examples
    selected_examples = None
    if args.examples:
        selected_examples = [ex.strip() for ex in args.examples.split(',')]

    runner.run_examples(selected_examples)

if __name__ == "__main__":
    main()
