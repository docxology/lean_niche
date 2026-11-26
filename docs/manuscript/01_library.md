# Library and exact models

The pinned Lean 4 compiler checks the default library and executable targets.
The library import root selects supported modules. The default build also
compiles the CLI and the executable regression suite. Lean supplies the
language and proof-checking framework [@lean4; @tpil].

The exact linear algebra module defines rational vectors and proves stated
identities for vector addition and scalar multiplication. Exact statistics
proves the singleton mean identity and ordered alternatives for two rational
interval endpoints. Their source files print dependency information for
the declarations.

Float computation APIs continue to use floating-point values. The
repository's executable counterexamples illustrate rounding, overflow, and
the behavior of NaN in comparisons. The examples motivate separate exact
models for algebraic proofs.
