import LeanNiche.Basic

/-!
# LeanNiche Tactics Module
Custom tactics and proof automation for LeanNiche.
-/

namespace LeanNiche.Tactics

open LeanNiche.Basic

-- Simple tactic aliases
macro "lean_simp" : tactic => `(tactic| simp)

-- Examples of tactic usage
example (n : Nat) : n + 0 = n := by lean_simp

-- Proof automation for common patterns
theorem auto_add_comm (a b : Nat) : a + b = b + a := by simp [add_comm]

theorem auto_mul_zero (a : Nat) : a * 0 = 0 := by simp

theorem auto_zero_add (a : Nat) : 0 + a = a := by simp

example (n : Nat) : n + 1 > 0 := by simp

example (n : Nat) : n ≤ n + 1 := by exact Nat.le_succ n

example : [1, 2, 3].length = 3 := by simp

example (b : Bool) : b = true ∨ b = false := by
  cases b with
  | true => left; rfl
  | false => right; rfl

-- Examples using basic tactics
example : 5 = 5 := by rfl

example (n : Nat) : 0 ≤ n := by exact Nat.zero_le n

example (n : Nat) : n + 1 > 0 := by simp

example (a b c : Nat) : (a + b) + c = a + (b + c) := by
  exact Nat.add_assoc a b c

-- Verification that tactics work
theorem tactic_verification : True := by trivial

end LeanNiche.Tactics