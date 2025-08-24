/- Canonical Basic module moved from core to LeanNiche namespace. -/

/-!
# Basic Mathematical Foundations

This module provides fundamental mathematical concepts and proofs that form the
basis for advanced topics in statistics and dynamical systems.
-/

namespace LeanNiche.Basic

/- Logical theorems for proof construction -/
theorem modus_ponens {p q : Prop} : (p → q) → p → q := by
  intro hpq hp
  exact hpq hp

/- Basic function properties -/
def injective {α β : Type} (f : α → β) : Prop :=
  ∀ x y : α, f x = f y → x = y

def surjective {α β : Type} (f : α → β) : Prop :=
  ∀ y : β, ∃ x : α, f x = y

/- Function composition -/
def composition {α β γ : Type} (f : β → γ) (g : α → β) : α → γ :=
  λ x => f (g x)

/- Identity function -/
def identity {α : Type} (x : α) : α := x

/- Basic arithmetic properties (Nat) -/
theorem add_comm (m n : Nat) : m + n = n + m := by
  induction m with
  | zero => rw [Nat.zero_add, Nat.add_zero]
  | succ m' ih => rw [Nat.succ_add, ih, ←Nat.add_succ]

end LeanNiche.Basic

/- Move core.Basic into LeanNiche namespace by copying its contents here.
   This file will be the canonical location for the Basic module. */
import core.Basic



