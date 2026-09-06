# src/lean/ — Lean 4 Source

The Lean 4 library for LeanNiche (lake lib `LeanNiche`, `srcDir = "src/lean"` per `lakefile.toml`).

## Contents
- `LeanNiche.lean`, `RootMain.lean` — module roots/imports
- `LeanNiche/` — consolidated module namespace (basic proofs, stats, linear algebra, dynamics, control, Lyapunov, etc.) plus `generated_artifacts/`
- `core/` — entry point and setup (`Main.lean` is the `lean_niche` executable root)
- `analysis/`, `algebra/`, `probability/`, `dynamics/`, `computation/` — domain modules
- `ai/` — active inference / free energy principle formalizations (in development)
- `utils/` — utilities and visualization

Build with `lake build`; run the main executable with `lake exe lean_niche`.
