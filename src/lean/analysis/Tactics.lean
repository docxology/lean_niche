/-!
# Tactics and Proofs

This file demonstrates basic proof tactics in Lean.
-/

namespace LeanNiche.Tactics

/-- Simple proof using tactics -/
example (p q : Prop) : p → (q → p) := by
  intro hp hq
  exact hp

/-- Simple function definition -/
def double (n : Nat) : Nat := 2 * n

/-- Simple proof about double -/
theorem double_zero : double 0 = 0 := by
  rfl

end LeanNiche.Tactics