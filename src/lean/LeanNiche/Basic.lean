/-!
# Basic Mathematical Foundations

This module provides fundamental mathematical concepts and proofs that form the
basis for advanced topics in statistics and dynamical systems.
-/

namespace LeanNiche.Basic

-- Logical theorems for proof construction
theorem modus_ponens {p q : Prop} : (p → q) → p → q := by
  intro hpq hp
  exact hpq hp

-- Basic function properties
def injective {α β : Type} (f : α → β) : Prop :=
  ∀ x y : α, f x = f y → x = y

def surjective {α β : Type} (f : α → β) : Prop :=
  ∀ y : β, ∃ x : α, f x = y

-- Function composition
def composition {α β γ : Type} (f : β → γ) (g : α → β) : α → γ :=
  fun x => f (g x)

-- Identity function
def identity {α : Type} (x : α) : α := x

-- Basic arithmetic properties
theorem add_comm (m n : Nat) : m + n = n + m := by
  induction m with
  | zero => rw [Nat.zero_add, Nat.add_zero]
  | succ m' ih => rw [Nat.succ_add, ih, ← Nat.add_succ]

-- Basic logical operations
theorem and_comm {p q : Prop} : p ∧ q ↔ q ∧ p := by
  constructor
  · intro ⟨hp, hq⟩
    exact ⟨hq, hp⟩
  · intro ⟨hq, hp⟩
    exact ⟨hp, hq⟩

theorem or_comm {p q : Prop} : p ∨ q ↔ q ∨ p := by
  constructor
  · intro h
    cases h with
    | inl hp => exact Or.inr hp
    | inr hq => exact Or.inl hq
  · intro h
    cases h with
    | inl hq => exact Or.inr hq
    | inr hp => exact Or.inl hp

-- Basic number theory
theorem zero_le (n : Nat) : 0 ≤ n := Nat.zero_le n

theorem succ_pos (n : Nat) : 0 < n + 1 := Nat.succ_pos n

end LeanNiche.Basic

