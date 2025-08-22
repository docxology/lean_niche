/-!
# Advanced Lean Environment Test

This file demonstrates more advanced Lean concepts in a simple working environment.
-/

namespace LeanNiche.Advanced

/-- Simple recursive function -/
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Simple proof about factorial -/
theorem factorial_zero : factorial 0 = 1 := by
  rfl

/-- Simple proof about factorial -/
theorem factorial_one : factorial 1 = 1 := by
  rfl

/-- Simple proof about factorial -/
theorem factorial_two : factorial 2 = 2 := by
  rfl

/-- Simple theorem about natural numbers -/
theorem add_zero (n : Nat) : n + 0 = n := by
  cases n with
  | zero => rfl
  | succ n' => exact congrArg Nat.succ (add_zero n')

/-- Simple theorem about lists -/
def list_reverse {α : Type} : List α → List α
  | [] => []
  | x :: xs => list_reverse xs ++ [x]

/-- Simple proof about list reversal -/
theorem reverse_nil {α : Type} : list_reverse ([] : List α) = [] := by
  rfl

end LeanNiche.Advanced