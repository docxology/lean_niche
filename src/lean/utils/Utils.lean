/-!
# Utility Functions and Helpers

This module provides general utility functions, mathematical helpers,
and common operations used throughout the LeanNiche environment.
-/

import LeanNiche.Basic

namespace LeanNiche.Utils

open LeanNiche.Basic

/-- Utility functions for working with natural numbers -/
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def binomial (n k : Nat) : Nat :=
  if k > n then 0 else
    factorial n / (factorial k * factorial (n - k))

def gcd (a b : Nat) : Nat :=
  match b with
  | 0 => a
  | _ => gcd b (a % b)

def lcm (a b : Nat) : Nat :=
  if a = 0 ∧ b = 0 then 0 else
    (a * b) / gcd a b

/-- Mathematical constants and common values -/
def pi_approx : Nat := 3141592653589793  -- π * 10^15 for precision
def e_approx : Nat := 2718281828459045   -- e * 10^15 for precision
def golden_ratio : Nat := 1618033988749895 -- φ * 10^15 for precision

/-- Utility functions for sequences and series -/
def arithmetic_sequence (a d : Nat) (n : Nat) : Nat :=
  a + n * d

def geometric_sequence (a r : Nat) (n : Nat) : Nat :=
  a * (r ^ n)

def sum_range (f : Nat → Nat) (start : Nat) (end_ : Nat) : Nat :=
  if start >= end_ then 0 else
    f start + sum_range f (start + 1) end_

/-- Utility functions for lists -/
def list_sum : List Nat → Nat
  | [] => 0
  | x :: xs => x + list_sum xs

def list_product : List Nat → Nat
  | [] => 1
  | x :: xs => x * list_product xs

def list_max : List Nat → Nat
  | [] => 0
  | [x] => x
  | x :: xs => Nat.max x (list_max xs)

def list_min : List Nat → Nat
  | [] => 0
  | [x] => x
  | x :: xs => Nat.min x (list_min xs)

def list_average (xs : List Nat) : Nat :=
  let sum := list_sum xs
  let len := xs.length
  if len = 0 then 0 else sum / len

def list_median (xs : List Nat) : Nat :=
  let sorted := xs.mergeSort (λ a b => a ≤ b)
  let len := sorted.length
  if len = 0 then 0 else
    if len % 2 = 1 then sorted.get! (len / 2) else
      let mid1 := sorted.get! (len / 2 - 1)
      let mid2 := sorted.get! (len / 2)
      (mid1 + mid2) / 2

/-- Utility functions for sets and relations -/
def powerset {α : Type} (s : List α) : List (List α) :=
  match s with
  | [] => [[]]
  | x :: xs =>
    let ps := powerset xs
    ps ++ ps.map (λ subset => x :: subset)

def cartesian_product {α β : Type} (xs : List α) (ys : List β) : List (α × β) :=
  xs.bind (λ x => ys.map (λ y => (x, y)))

/-- Mathematical utility functions -/
def abs_diff (a b : Nat) : Nat :=
  if a > b then a - b else b - a

def percentage (part whole : Nat) : Nat :=
  if whole = 0 then 0 else (part * 100) / whole

def percentage_of (value : Nat) (percentage : Nat) : Nat :=
  (value * percentage) / 100

def clamp (value min max : Nat) : Nat :=
  if value < min then min else
    if value > max then max else value

/-- Random number generation utilities (deterministic) -/
def linear_congruential_generator (seed : Nat) : Nat → Nat
  | 0 => seed
  | n + 1 =>
    let prev := linear_congruential_generator seed n
    (prev * 1103515245 + 12345) % (2^31)

def pseudo_random_sequence (seed : Nat) (len : Nat) : List Nat :=
  List.range len |>.map (linear_congruential_generator seed)

/-- String and formatting utilities -/
def nat_to_string : Nat → String
  | 0 => "0"
  | n =>
    let rec digits : Nat → List Nat → List Nat
      | 0, acc => acc
      | m, acc => digits (m / 10) ((m % 10) :: acc)
    let digit_list := digits n []
    let digit_chars := digit_list.map (λ d =>
      match d with
      | 0 => '0'
      | 1 => '1'
      | 2 => '2'
      | 3 => '3'
      | 4 => '4'
      | 5 => '5'
      | 6 => '6'
      | 7 => '7'
      | 8 => '8'
      | 9 => '9'
      | _ => '?'
    )
    String.mk digit_chars

def format_percentage (value : Nat) : String :=
  let integer_part := value / 100
  let decimal_part := value % 100
  nat_to_string integer_part ++ "." ++
  (if decimal_part < 10 then "0" ++ nat_to_string decimal_part else nat_to_string decimal_part) ++ "%"

/-- Error handling and validation utilities -/
def validate_positive (n : Nat) : Option Nat :=
  if n > 0 then some n else none

def validate_range (n min max : Nat) : Option Nat :=
  if n ≥ min ∧ n ≤ max then some n else none

def validate_probability (p : Nat) : Option Nat :=
  if p ≤ 1000 then some p else none  -- p in [0, 1000] for per mille

/-- Debugging and logging utilities -/
def debug_print (message : String) (value : Nat) : String :=
  message ++ ": " ++ nat_to_string value

def format_result (operation : String) (input : Nat) (output : Nat) : String :=
  operation ++ "(" ++ nat_to_string input ++ ") = " ++ nat_to_string output

def format_list (name : String) (xs : List Nat) : String :=
  name ++ " = [" ++
  String.intercalate ", " (xs.map nat_to_string) ++
  "]"

end LeanNiche.Utils
