import LeanNiche

/-! Executable controls for the supported computation API and exact proof model. -/
private def require (condition : Bool) (message : String) : IO Unit :=
  unless condition do throw (IO.userError message)

example {n : Nat} (u v w : LeanNiche.ExactLinearAlgebra.Vector n) :
    LeanNiche.ExactLinearAlgebra.vector_add
      (LeanNiche.ExactLinearAlgebra.vector_add u v) w =
    LeanNiche.ExactLinearAlgebra.vector_add u
      (LeanNiche.ExactLinearAlgebra.vector_add v w) :=
  LeanNiche.ExactLinearAlgebra.vector_add_assoc u v w

example (x : Rat) : LeanNiche.ExactStatistics.sample_mean ⟨1, [x]⟩ = x :=
  LeanNiche.ExactStatistics.sample_mean_singleton x

def main : IO Unit := do
  require (LeanNiche.Utils.safe_divide 6.0 2.0 == some 3.0)
    "finite division changed"
  require (LeanNiche.Utils.safe_divide 6.0 0.0 == none)
    "zero denominator must reject"
  require (LeanNiche.Statistics.sample_mean ⟨2, [2.0, 4.0]⟩ == 3.0)
    "Float sample mean changed"
  let v : LeanNiche.LinearAlgebra.Vector 1 := fun _ => 2.0
  let w : LeanNiche.LinearAlgebra.Vector 1 := fun _ => 3.0
  require (LeanNiche.LinearAlgebra.vector_add v w 0 == 5.0)
    "Float vector addition changed"
  let a : Float := 10000000000000000.0
  let b : Float := -10000000000000000.0
  let c : Float := 1.0
  require (!((a + b) + c == a + (b + c)))
    "rounding counterexample unexpectedly disappeared"
  let nan : Float := 0.0 / 0.0
  require (!(decide (nan ≤ 1.0) || decide (nan > 1.0)))
    "NaN must not be treated as an ordered rational"
  IO.println "LeanNiche: computation controls and exact proof consumers passed"
