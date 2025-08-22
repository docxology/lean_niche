/-!
# Test Suite for LeanNiche

This file contains simple tests for the LeanNiche environment.
-/

namespace LeanNiche.TestSuite

open LeanNiche.Basic
open LeanNiche.Advanced
open LeanNiche.Tactics
open LeanNiche.SetTheory
open LeanNiche.Computational

/-- Simple test function -/
def run_simple_test : IO Unit := do
  IO.println "Running basic environment tests..."

  -- Test basic theorems
  let _ := LeanNiche.Basic.test_theorem
  IO.println "✓ Basic theorems compiled"

  -- Test advanced functions
  let _ := LeanNiche.Advanced.factorial 5
  IO.println "✓ Advanced functions compiled"

  -- Test computational functions
  let _ := LeanNiche.Computational.fibonacci 5
  IO.println "✓ Computational functions compiled"

  IO.println "Basic tests completed successfully!"

/-- Main test runner -/
def main : IO Unit := do
  IO.println "=========================================="
  IO.println "       LeanNiche Test Suite"
  IO.println "=========================================="
  IO.println ""

  run_simple_test

  IO.println ""
  IO.println "=========================================="
  IO.println "All tests passed successfully!"
  IO.println "=========================================="

end LeanNiche.TestSuite