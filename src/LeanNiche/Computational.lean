/-!
# Computational Mathematics

This file demonstrates computational algorithms in Lean.
-/

namespace LeanNiche.Computational

/-- Simple recursive function -/
def fibonacci : Nat → Nat
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

/-- Simple proofs about fibonacci -/
theorem fibonacci_zero : fibonacci 0 = 0 := by
  rfl

theorem fibonacci_one : fibonacci 1 = 1 := by
  rfl

/-- Simple list operations -/
def list_length {α : Type} : List α → Nat
  | [] => 0
  | _ :: xs => 1 + list_length xs

/-- Simple proof about list length -/
theorem length_nil {α : Type} : list_length ([] : List α) = 0 := by
  rfl

end LeanNiche.Computational