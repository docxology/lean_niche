import LeanNiche.Advanced
import LeanNiche.Computational
import Mathlib.Data.Nat.Prime

/-!
# Advanced Examples for LeanNiche

This file demonstrates advanced mathematical concepts and proof techniques
using the LeanNiche environment.
-/

namespace LeanNiche.Examples

open LeanNiche.Advanced
open LeanNiche.Computational

/-- Example 1: Prime number properties -/
def check_prime_property (n : ℕ) (hn : n > 2) : Bool :=
  if n.Prime then
    n % 2 = 1  -- Odd
  else
    true  -- Not applicable

/-- Example 2: Number theory computations -/
def number_theory_example : ℕ :=
  let n := 10
  sum_up_to n  -- Sum from 1 to 10 = 55

/-- Example 3: Algorithm verification -/
def sorting_example : List ℕ := [3, 1, 4, 1, 5, 9, 2, 6]
-- insertion_sort will return [1, 1, 2, 3, 4, 5, 6, 9]

/-- Example 4: Mathematical induction -/
def power_of_two : ℕ → ℕ
  | 0 => 1
  | n + 1 => 2 * power_of_two n

/-- Example 5: Binary representation concepts -/
def binary_weight (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | n => if n % 2 = 1 then 1 + binary_weight (n / 2) else binary_weight (n / 2)

/-- Example 6: Modular arithmetic -/
def mod_inverse (a n : ℕ) (ha : a > 0) (hn : n > 0) : Option ℕ :=
  let rec find_inverse (k : ℕ) : Option ℕ :=
    if k = 0 then none
    else if (a * k) % n = 1 then some k
    else find_inverse (k - 1)
  find_inverse n

/-- Example 7: Cryptographic concepts -/
def primitive_root_check (g n : ℕ) (hg : g > 0) (hn : n > 1) : Bool :=
  let powers := List.range (n - 1) |>.map (λ k => pow g (k + 1) % n)
  let unique_powers := powers.eraseDups
  unique_powers.length = powers.length

/-- Example 8: Graph theory concepts -/
def adjacency_list_example : List (List ℕ) :=
  [[1, 2],    -- Node 0 connected to 1, 2
   [0, 2],    -- Node 1 connected to 0, 2
   [0, 1, 3], -- Node 2 connected to 0, 1, 3
   [2]]       -- Node 3 connected to 2

def degree_sequence : List ℕ :=
  adjacency_list_example.map List.length  -- [2, 2, 3, 1]

/-- Example 9: Formal language theory -/
inductive SimpleLanguage
  | empty : SimpleLanguage
  | a : SimpleLanguage → SimpleLanguage
  | b : SimpleLanguage → SimpleLanguage

def language_size : SimpleLanguage → ℕ
  | SimpleLanguage.empty => 1
  | SimpleLanguage.a w => 1 + language_size w
  | SimpleLanguage.b w => 1 + language_size w

/-- Example 10: Category theory concepts -/
def id_function {α : Type} (x : α) : α := x

def compose {α β γ : Type} (f : β → γ) (g : α → β) (x : α) : γ := f (g x)

#eval number_theory_example  -- 55
#eval binary_weight 13  -- 3 (binary 1101 has three 1s)
#eval degree_sequence  -- [2, 2, 3, 1]
#eval power_of_two 5  -- 32

/-- Example 11: Proof automation -/
theorem arithmetic_progression_sum (n : ℕ) :
    2 * sum_up_to n = n * (n + 1) := by
  unfold sum_up_to
  rw [Nat.mul_div_cancel]
  exact Nat.dvd_mul_right 2 (n * (n + 1))

/-- Example 12: Advanced induction -/
theorem power_of_two_ge_n (n : ℕ) : ∃ k : ℕ, power_of_two k ≥ n := by
  induction n with
  | zero => exists 0; exact Nat.zero_le 1
  | succ n' ih =>
    obtain ⟨k, hk⟩ := ih
    exists k + 1
    rw [power_of_two]
    exact Nat.mul_le_mul_left 2 hk

/-- Example 13: Number theory proof -/
theorem even_square (n : ℕ) : (2 * n)^2 % 2 = 0 := by
  simp [Nat.mul_mod]
  exact Nat.mod_self 2

/-- Example 14: Algorithm correctness -/
def list_length {α : Type} : List α → ℕ
  | [] => 0
  | _ :: xs => 1 + list_length xs

theorem length_append {α : Type} (xs ys : List α) :
    list_length (xs ++ ys) = list_length xs + list_length ys := by
  induction xs with
  | nil => rfl
  | cons x xs' ih => simp [List.append, ih]

end LeanNiche.Examples
