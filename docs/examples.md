# Examples

Start with `lake exe test_suite` and `lake env lean FloatCounterexamples.lean`
from the [setup guide](deployment.md). These run the repository's checked
regression executable and its separate Float counterexample controls.

Use the current Python CLI help to discover analysis and plotting commands.

```sh
uv run --no-sync --python 3.13 lean-niche --help
```

The Python example sources are in [examples/](../examples/README.md).
Run an example from a private output directory to keep its results separate
from the retained repository artifacts. Python numerical results and plots
illustrate the computations. A theorem claim requires its own compiler and
axiom report for the submitted source.

The [retained example execution report](../outputs/examples_execution_report.md)
describes its original run. Re-run the relevant case to obtain a current result.
