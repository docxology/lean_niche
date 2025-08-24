/-!
# Tests for ProofExport (Lean side)
-/

import Lean
import LeanNiche.ProofExport

open Lean
open LeanNiche.ProofExport

def main : IO Unit := do
  let opts : Options := {}
  let (env, _) ← importModules #[] opts
  -- Basic smoke tests: export code and query a known constant if available
  let exported := (exportEnvironmentCode env).run' { env := env }
  if exported.length > 0 then
    IO.println "✅ exportEnvironmentCode produced output"
  else
    IO.println "⚠️  exportEnvironmentCode produced empty output"

  -- Query a theorem likely present in core (Nat.add_comm)
  let thmName := Name.str (Name.str Name.anonymous "Nat") "add_comm"
  let proof := (proofOutputFor env thmName).run' { env := env }
  IO.println proof


