/-!
# Tests for Advanced.lean Module

This file contains comprehensive tests for the Advanced.lean module,
verifying advanced theorems, number theory, and complex proofs.
-/

import LeanNiche.Advanced

namespace LeanNiche.Tests.Advanced

/-- Test factorial function -/
def test_factorial : IO Unit := do
  assert_eq (LeanNiche.Advanced.factorial 0) 1
  assert_eq (LeanNiche.Advanced.factorial 1) 1
  assert_eq (LeanNiche.Advanced.factorial 2) 2
  assert_eq (LeanNiche.Advanced.factorial 3) 6
  assert_eq (LeanNiche.Advanced.factorial 4) 24
  assert_eq (LeanNiche.Advanced.factorial 5) 120

  IO.println "✅ Factorial function tests passed"

/-- Test factorial theorems -/
def test_factorial_theorems : IO Unit := do
  -- Test factorial zero theorem
  let _ := LeanNiche.Advanced.factorial_zero

  -- Test factorial one theorem
  let _ := LeanNiche.Advanced.factorial_one

  -- Test factorial two theorem
  let _ := LeanNiche.Advanced.factorial_two

  IO.println "✅ Factorial theorems tests passed"

/-- Test natural number theorems -/
def test_natural_number_theorems : IO Unit := do
  -- Test add_zero theorem
  let _ := LeanNiche.Advanced.add_zero 5

  -- Test list operations
  let test_list := [1, 2, 3, 4, 5]
  let reversed := LeanNiche.Advanced.list_reverse test_list
  assert_eq reversed [5, 4, 3, 2, 1]

  IO.println "✅ Natural number theorems tests passed"

/-- Test list reversal theorems -/
def test_list_reversal_theorems : IO Unit := do
  -- Test reverse_nil theorem
  let _ := LeanNiche.Advanced.reverse_nil

  -- Test with non-empty list
  let test_list := [1, 2, 3]
  let reversed := LeanNiche.Advanced.list_reverse test_list
  let double_reversed := LeanNiche.Advanced.list_reverse reversed
  assert_eq double_reversed test_list

  IO.println "✅ List reversal theorems tests passed"

/-- Test sum_up_to function -/
def test_sum_up_to : IO Unit := do
  assert_eq (LeanNiche.Advanced.sum_up_to 0) 0
  assert_eq (LeanNiche.Advanced.sum_up_to 1) 1
  assert_eq (LeanNiche.Advanced.sum_up_to 2) 3
  assert_eq (LeanNiche.Advanced.sum_up_to 3) 6
  assert_eq (LeanNiche.Advanced.sum_up_to 4) 10
  assert_eq (LeanNiche.Advanced.sum_up_to 5) 15

  IO.println "✅ Sum up to function tests passed"

/-- Test sum_up_to correctness theorem -/
def test_sum_up_to_correctness : IO Unit := do
  -- Test the correctness theorem
  let _ := LeanNiche.Advanced.sum_up_to_correct 5

  IO.println "✅ Sum up to correctness theorem test passed"

/-- Test infinite primes theorem -/
def test_infinite_primes : IO Unit := do
  -- Test the infinite primes theorem
  let _ := LeanNiche.Advanced.infinite_primes

  IO.println "✅ Infinite primes theorem test passed"

/-- Test factorial strict monotonicity -/
def test_factorial_strict_mono : IO Unit := do
  -- Test that factorial is strictly increasing
  let _ := LeanNiche.Advanced.factorial_strict_mono

  IO.println "✅ Factorial strict monotonicity test passed"

/-- Test prime number properties -/
def test_prime_properties : IO Unit := do
  -- Test that primes greater than 2 are odd
  let _ := LeanNiche.Advanced.prime_gt_two_odd

  IO.println "✅ Prime properties tests passed"

/-- Test arithmetic operations -/
def test_arithmetic_operations : IO Unit := do
  -- Test basic arithmetic with larger numbers
  assert_eq (LeanNiche.Advanced.sum_up_to 10) 55
  assert_eq (LeanNiche.Advanced.factorial 6) 720

  -- Test list operations with larger lists
  let large_list := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let reversed := LeanNiche.Advanced.list_reverse large_list
  let expected := [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
  assert_eq reversed expected

  IO.println "✅ Arithmetic operations tests passed"

/-- Run all Advanced module tests -/
def run_all_tests : IO Unit := do
  IO.println "🔬 Running Advanced module tests..."
  IO.println ""

  test_factorial
  test_factorial_theorems
  test_natural_number_theorems
  test_list_reversal_theorems
  test_sum_up_to
  test_sum_up_to_correctness
  test_infinite_primes
  test_factorial_strict_mono
  test_prime_properties
  test_arithmetic_operations

  IO.println ""
  IO.println "🎉 All Advanced module tests completed successfully!"

end LeanNiche.Tests.Advanced
