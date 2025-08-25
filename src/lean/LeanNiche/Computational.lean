/-!
# LeanNiche Computational Module
Computational methods and algorithms for LeanNiche.
-/

namespace LeanNiche

/-- Computational structure for numerical methods -/
structure Computational where
  precision : Nat
  iterations : Nat

/-- Simple iterative computation -/
def iterative_sum (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | n + 1 => n + 1 + iterative_sum n

/-- Simple computational theorem -/
theorem iterative_sum_base : iterative_sum 0 = 0 := by
  rfl

end LeanNiche
