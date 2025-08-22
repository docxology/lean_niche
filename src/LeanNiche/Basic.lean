/-!
# Basic Lean Environment Test

This file demonstrates a simple working Lean environment without external dependencies.
-/

namespace LeanNiche.Basic

/-- A simple theorem to test the environment -/
theorem test_theorem : ∀ (p q : Prop), p → (q → p) := by
  intro p q hp hq
  exact hp

/-- Simple function definition -/
def identity {α : Type} (x : α) : α := x

/-- Simple proof using identity -/
theorem identity_property {α : Type} (x : α) : identity x = x := by
  rfl

/-- Basic logical equivalence -/
def logical_implication (p q : Prop) : Prop := p → q

/-- Simple list operations -/
def list_length {α : Type} : List α → Nat
  | [] => 0
  | _ :: xs => 1 + list_length xs

/-- Length property -/
theorem list_length_nil {α : Type} : list_length ([] : List α) = 0 := by
  rfl

end LeanNiche.Basic