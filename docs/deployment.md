# Set up and verify LeanNiche

Install Git, uv, and Lean with an Elan-managed toolchain or a matching
standalone distribution. Use the Lean version recorded in
[`lean-toolchain`](../lean-toolchain). The local workflow uses Python 3.13
and the committed uv lockfile.

From the checkout root, run the compiler, its regression executable, and the
Float counterexample file.

```sh
uv sync --locked --python 3.13
lean --version
lake --wfail build
lake exe test_suite
lake env lean FloatCounterexamples.lean
```

Then run the Python suite and the compiler integration battery.

```sh
uv run --no-sync --python 3.13 python -m pytest src/tests/python/
mkdir -p outputs/verification
uv run --no-sync --python 3.13 python src/tests/run_comprehensive_lean_tests.py --output-file outputs/verification/comprehensive_test_report.json
```

Run Python with `-m pytest` to put the checkout root on the module search path.
The Python checks that exercise Lean need the compiler on `PATH`. The
comprehensive report is a local result for the current input and toolchain.

The [methods manuscript build](manuscript-build.md) uses a separate renderer
environment. The native Lean and Python checks establish the behavior the
manuscript describes.
