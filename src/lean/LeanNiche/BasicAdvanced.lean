/- Canonical Advanced module moved from core to LeanNiche namespace (copied).

namespace LeanNiche.Advanced

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_zero : factorial 0 = 1 := by rfl
theorem factorial_one : factorial 1 = 1 := by rfl
theorem factorial_two : factorial 2 = 2 := by rfl

theorem add_zero (n : Nat) : n + 0 = n := by
  cases n with
  | zero => rfl
  | succ n' => exact congrArg Nat.succ (add_zero n')

def list_reverse {α : Type} : List α → List α
  | [] => []
  | x :: xs => list_reverse xs ++ [x]

theorem reverse_nil {α : Type} : list_reverse ([] : List α) = [] := by rfl

end LeanNiche.Advanced

import analysis.Advanced

/- Wrapper forwarding to `analysis.Advanced` -/


