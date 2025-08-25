/-!
# LeanNiche Utils Module
Utility functions and helpers for LeanNiche.
-/

namespace LeanNiche

/-- Utility function for maximum -/
def max_nat (x y : Nat) : Nat :=
  if x ≥ y then x else y

/-- Simple utility theorem -/
theorem max_exists : ∀ x y : Nat, max_nat x y = x ∨ max_nat x y = y := by
  intro x y
  by_cases h : x ≥ y
  · left
    simp [max_nat, if_pos h]
  · right
    simp [max_nat, if_neg h]

end LeanNiche
