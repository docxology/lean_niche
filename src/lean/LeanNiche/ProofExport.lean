/- Canonical ProofExport moved to LeanNiche namespace (copied from core).

import Lean

open Lean

namespace LeanNiche.ProofExport

def renderConstant (n : Name) (ci : ConstantInfo) : CoreM (Option String) := do
  match ci with
  | .defnInfo d =>
    let ty ← ppExpr d.type |>.run
    let val ← ppExpr d.value |>.run
    pure <| some s!"def {n} : {ty} := {val}\n"
  | .thmInfo t =>
    let ty ← ppExpr t.type |>.run
    let val ← ppExpr t.value |>.run
    pure <| some s!"theorem {n} : {ty} := {val}\n"
  | _ => pure none

def exportEnvironmentCode (env : Environment) : CoreM String := do
  let mut acc := String.empty
  for (n, ci) in env.constants.toList do
    let some s ← renderConstant n ci | pure ()
    acc := acc ++ s ++ "\n"
  pure acc

def proofOutputFor (env : Environment) (thmName : Name) : CoreM String := do
  match env.find? thmName with
  | some (.thmInfo t) =>
    let ty ← ppExpr t.type |>.run
    let val ← ppExpr t.value |>.run
    pure s!"Proof of {thmName}:\nType: {ty}\nTerm: {val}\n"
  | some ci => pure s!"{thmName} is not a theorem"
  | none => pure s!"Theorem {thmName} not found"

def main (argv : List String) : IO Unit := do
  let opts : Options := {}
  let importList : Array Import := #[]
  let (env, _) ← importModules importList opts
  match argv with
  | ["--export"] => IO.println (exportEnvironmentCode env).run' { env := env }
  | ["--theorem", nm] => IO.println (proofOutputFor env (Name.mkSimple nm)).run' { env := env }
  | _ => IO.println "Usage: --export | --theorem <Name>"

end LeanNiche.ProofExport


