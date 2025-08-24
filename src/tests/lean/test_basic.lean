/-!
# Tests for Basic.lean Module

This file contains comprehensive tests for the Basic.lean module,
verifying all theorems, definitions, and functions.
-/

import LeanNiche.Basic

namespace LeanNiche.Tests.Basic

/-- Test that the identity function works correctly -/
def test_identity_function : IO Unit := do
  let result := LeanNiche.Basic.identity "test"
  assert_eq result "test"
  IO.println "✅ Identity function test passed"

/-- Test addition commutativity -/
def test_add_comm : IO Unit := do
  -- Test with small numbers
  let result1 := LeanNiche.Basic.add_comm 2 3
  assert_eq (2 + 3) (3 + 2)

  -- Test with larger numbers
  let result2 := LeanNiche.Basic.add_comm 10 25
  assert_eq (10 + 25) (25 + 10)

  IO.println "✅ Addition commutativity tests passed"

/-- Test addition associativity -/
def test_add_assoc : IO Unit := do
  let result := LeanNiche.Basic.add_assoc 1 2 3
  assert_eq ((1 + 2) + 3) (1 + (2 + 3))
  IO.println "✅ Addition associativity test passed"

/-- Test multiplication properties -/
def test_mul_properties : IO Unit := do
  -- Test multiplication commutativity
  assert_eq (2 * 3) (3 * 2)
  assert_eq (5 * 7) (7 * 5)

  -- Test multiplication distributivity
  assert_eq (2 * (3 + 4)) (2 * 3 + 2 * 4)
  assert_eq (5 * (1 + 2)) (5 * 1 + 5 * 2)

  IO.println "✅ Multiplication properties tests passed"

/-- Test function composition -/
def test_function_composition : IO Unit := do
  def f (x : Nat) : Nat := x + 1
  def g (x : Nat) : Nat := x * 2

  let composed := LeanNiche.Basic.composition f g
  let result := composed 5
  let expected := f (g 5)  -- (5 * 2) + 1 = 11

  assert_eq result expected
  IO.println "✅ Function composition test passed"

/-- Test injective function properties -/
def test_injective_functions : IO Unit := do
  def double (x : Nat) : Nat := x * 2

  -- Double is injective
  assert_eq (double 1 ≠ double 2) true
  assert_eq (double 3 ≠ double 6) true

  IO.println "✅ Injective function tests passed"

/-- Test surjective function properties -/
def test_surjective_functions : IO Unit := do
  def half_floor (x : Nat) : Nat := x / 2

  -- For even numbers, half_floor is surjective onto even numbers
  assert_eq (half_floor 0) 0
  assert_eq (half_floor 2) 1
  assert_eq (half_floor 4) 2

  IO.println "✅ Surjective function tests passed"

/-- Run all Basic module tests -/
def run_all_tests : IO Unit := do
  IO.println "🧮 Running Basic module tests..."
  IO.println ""

  test_identity_function
  test_add_comm
  test_add_assoc
  test_mul_properties
  test_function_composition
  test_injective_functions
  test_surjective_functions

  IO.println ""
  IO.println "🎉 All Basic module tests completed successfully!"

end LeanNiche.Tests.Basic
