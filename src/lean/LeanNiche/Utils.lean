import LeanNiche.Basic

/-!
# LeanNiche Utils Module
Utility functions and helper definitions.
-/

namespace LeanNiche.Utils

open LeanNiche.Basic

-- Type aliases for clarity
def Real := Float
def Natural := Nat
def Integer := Int

-- Utility functions
def clamp (x min_val max_val : Float) : Float :=
  if x < min_val then min_val
  else if x > max_val then max_val
  else x

def lerp (a b t : Float) : Float :=
  a + t * (b - a)

def sign (x : Float) : Float :=
  if x > 0.0 then 1.0
  else if x < 0.0 then -1.0
  else 0.0

-- List utilities
def list_average (xs : List Float) : Float :=
  if xs.isEmpty then 0.0 else xs.sum / xs.length.toFloat

-- Range generation
def float_range (start stop : Float) (steps : Nat) : List Float :=
  if steps = 0 then []
  else
    let step_size := (stop - start) / (steps - 1).toFloat
    (List.range steps).map (fun i => start + i.toFloat * step_size)

-- Utility theorems
theorem sign_cases (x : Float) :
  sign x = 1.0 ∨ sign x = 0.0 ∨ sign x = -1.0 := by
  simp [sign]
  by_cases h1 : x > 0.0
  · left; simp [h1]
  · by_cases h2 : x < 0.0
    · right; right; simp [h1, h2]
    · right; left; simp [h1, h2]

-- Error handling utilities
def safe_divide (a b : Float) : Option Float :=
  if b == 0.0 then none else some (a / b)

def safe_sqrt (x : Float) : Option Float :=
  if x < 0.0 then none else some (Float.sqrt x)

theorem safe_divide_some (a b : Float) (h : b ≠ 0.0) :
  ∃ result, safe_divide a b = some result := by
  sorry

end LeanNiche.Utils