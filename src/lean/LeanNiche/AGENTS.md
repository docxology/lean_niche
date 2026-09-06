# LeanNiche Core Library

**Layout**: Advanced.lean, Basic.lean, computational modules, domain-specific modules, generated_artifacts/
**Basic/Advanced**: Core mathematical foundations
**Computational**: ComputationalTests.lean, WorkingComputations.lean
**Domain modules**: ControlTheory.lean, DynamicalSystems.lean, Statistics.lean, etc.
**Generated**: generated_artifacts/ contains machine-generated proofs
**Main entry**: LeanNicheMain.lean, Main.lean for library coordination
**Utilities**: Setup.lean, Tactics.lean, Utils.lean, Visualization.lean

Primary library entry point. See README.md for module dependencies.
Verify: `lake build` builds entire library successfully.
