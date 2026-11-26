# Develop and check a proof

Build the library using the [verification guide](deployment.md). The project
keeps exact rational statements separate from floating-point computations.

To inspect the executed proofs in the exact statistics module, use the
pinned Lean compiler through Lake.

```sh
lake env lean src/lean/LeanNiche/ExactStatistics.lean
```

The source proves the singleton rational sample mean and the stated ordering
property for the interval endpoints. Its `#print axioms` lines report the
dependencies of those declarations. Read both the statement and its
dependencies before interpreting a successful compile.

The [Float counterexamples](../FloatCounterexamples.lean) are executable
controls. Their observations do not establish rational identities for all
Float values. See the [exact-model guide](exact-models.md) for that boundary.
