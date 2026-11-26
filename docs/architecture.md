# LeanNiche architecture

| Component | Role |
|---|---|
| [Lake targets](../lakefile.toml) | The default build checks the library, the `lean_niche` executable, and the `test_suite` executable. |
| [Library imports](../src/lean/LeanNiche.lean) | The public root imports the supported exact-model and computation modules. |
| [Exact linear algebra](../src/lean/LeanNiche/ExactLinearAlgebra.lean) and [exact statistics](../src/lean/LeanNiche/ExactStatistics.lean) | Rational models with checked identities and printed axiom dependencies. |
| [Python Lean runner](../src/python/core/lean_runner.py) | Invokes Lean, checks compiler and axiom results, and writes proof-evidence reports. |
| [Python CLI](../src/python/cli.py) | Analysis and plotting entry points exposed through the Python package. |
| [Example outputs](../outputs/README.md) | Generated or retained reports and proof-shaped files with separate evidence status. |

The Lake manifest has no external packages. The default import list defines
the checked library boundary. Other files and retained proof drafts need
their own compilation evidence before an individual claim can be treated as
verified. See [exact-model boundaries](exact-models.md).
