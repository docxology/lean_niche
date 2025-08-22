import LeanNiche.Basic
import LeanNiche.Advanced

/-!
# Basic Examples for LeanNiche

This file demonstrates how to use the LeanNiche environment for basic
mathematical proofs and computations.
-/

namespace LeanNiche.Examples

open LeanNiche.Basic
open LeanNiche.Advanced

/-- Example 1: Using basic arithmetic theorems -/
def example1 : ℕ :=
  let a := 3
  let b := 5
  let result := a + b
  -- We know from add_comm that result = b + a = 5 + 3 = 8
  result

/-- Example 2: Computing with factorials -/
def example2 : ℕ :=
  let n := 5
  let fact_n := factorial n  -- 5! = 120
  let sum_up_to_n := sum_up_to n  -- sum from 1 to 5 = 15
  fact_n + sum_up_to_n  -- 120 + 15 = 135

/-- Example 3: Working with Fibonacci numbers -/
def fibonacci_sequence : List ℕ :=
  [fibonacci 0, fibonacci 1, fibonacci 2, fibonacci 3, fibonacci 4, fibonacci 5]
  -- [0, 1, 1, 2, 3, 5]

/-- Example 4: Simple theorem application -/
theorem example4 (a b : ℕ) : a + b + 0 = a + b := by
  rw [Nat.add_zero]

/-- Example 5: Using advanced theorems -/
theorem example5 (n : ℕ) (hn : n > 0) : n! > 0 := by
  exact factorial_pos n

/-- Example 6: Computational verification -/
def verify_arithmetic : Bool :=
  let x := 2 + 3 * 4  -- 2 + 12 = 14
  let y := 2 * 3 + 2 * 4  -- 6 + 8 = 14
  x = y  -- Should be true by distributivity

/-- Example 7: List operations with verification -/
def list_example : List ℕ := [1, 2, 3, 4, 5]

def list_sum (xs : List ℕ) : ℕ :=
  match xs with
  | [] => 0
  | x :: xs' => x + list_sum xs'

/-- Example 8: Set operations -/
def example_set : Set ℕ := {n : ℕ | n % 2 = 0}  -- Even numbers

/-- Example 9: Function composition -/
def double (n : ℕ) : ℕ := 2 * n
def square (n : ℕ) : ℕ := n * n

def double_then_square (n : ℕ) : ℕ := square (double n)  -- 2n²

/-- Example 10: Proof by computation -/
def concrete_computation : ℕ :=
  let n := 3
  fibonacci (n + 2)  -- fibonacci 5 = 5

#eval example1  -- Should be 8
#eval example2  -- Should be 135
#eval fibonacci_sequence  -- Should be [0, 1, 1, 2, 3, 5]
#eval verify_arithmetic  -- Should be true
#eval list_sum list_example  -- Should be 15
#eval double_then_square 5  -- Should be 50
#eval concrete_computation  -- Should be 5

end LeanNiche.Examples
