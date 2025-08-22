import LeanNiche.Basic
import LeanNiche.Advanced
import LeanNiche.Tactics
import LeanNiche.SetTheory
import LeanNiche.Computational
import LeanNiche.TestSuite

/-!
# LeanNiche - Deep Research Environment

This is the main executable for the LeanNiche deep research environment.
It demonstrates various aspects of formal mathematics in Lean 4.

Features:
- Basic mathematical proofs
- Advanced theorem proving
- Computational algorithms
- Set theory and logic
- Automated proof tactics
- Comprehensive test suite
- Performance analysis
-/

def main : IO Unit := do
  IO.println "=========================================="
  IO.println "       LeanNiche Deep Research Environment"
  IO.println "=========================================="
  IO.println ""
  IO.println "Welcome to LeanNiche - A comprehensive Lean 4 research environment!"
  IO.println ""
  IO.println "This environment includes:"
  IO.println "• Mathematical proofs and theorems"
  IO.println "• Algorithm verification"
  IO.println "• Set theory and logic"
  IO.println "• Automated tactics"
  IO.println "• Computational mathematics"
  IO.println "• Comprehensive testing"
  IO.println ""
  IO.println "=========================================="
  IO.println ""

  -- Run the test suite
  LeanNiche.TestSuite.main

  IO.println ""
  IO.println "=========================================="
  IO.println "Environment Status: All systems operational"
  IO.println "=========================================="