# Float computation and exact proofs

The `LeanNiche.LinearAlgebra`, `Statistics` and `Utils` computation functions
retain their Float types and definitions. IEEE arithmetic has rounding, signed
zero, infinity and NaN; it is not a field or total order.

The former admitted vector algebra, singleton-mean and interval-order claims
are removed from the Float namespaces. Their exact counterparts live in
`LeanNiche.ExactLinearAlgebra` and `LeanNiche.ExactStatistics` over `Rat`.
All seven exact theorems compile and their retained `#print axioms` commands
report only `propext`, `Classical.choice` and `Quot.sound`. No admitted proof
is used in the default library. This is a proof-API correction; computation
APIs are unchanged.

`Utils.safe_divide_some` now states the exact Bool guard used by the function,
`(b == 0.0) = false`, instead of assuming propositional inequality is the
same as IEEE comparison. It is proved by the actual branch condition.

Executable witnesses in `FloatCounterexamples.lean` demonstrate rounded
associativity (1 versus 0), distributivity with overflow (0 versus NaN), and
the failure of ordinary order trichotomy for NaN. Those observations are
computational counterexamples, not a theorem about all Float executions.

## Runtime evidence

The Python runner rejects compiler failures, admissions, and theorem dependencies
outside `propext`, `Classical.choice`, and `Quot.sound`. Proof reports are bound
to the submitted source and Lean's `#print axioms` output. Discovered output
files are SHA-256 inventoried as `unverified_source`; reporting never copies
them into the library or runs `lake update`. Algorithm test cases compile with
the algorithm and execute an assertion against each expected value. Failed
compilations retain their diagnostic reports and report zero verified proofs.

CI acceptance requires the strict default build, native regression executable,
Float counterexamples, all Python tests and the complete eight-case runtime
compiler suite. Example commands run with private output directories; their
computational outputs do not certify the legacy research proof drafts.
