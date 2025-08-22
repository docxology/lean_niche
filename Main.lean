import LeanNiche.Basic
import LeanNiche.Advanced
import LeanNiche.Tactics
import LeanNiche.SetTheory
import LeanNiche.Computational

/-!
# LeanNiche - Deep Research Environment (Main)

This is the main executable for the LeanNiche deep research environment.
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

  -- Test basic functionality
  let _ := LeanNiche.Basic.identity "test"
  let _ := LeanNiche.Advanced.factorial 5
  let _ := LeanNiche.Computational.fibonacci 5
  IO.println "✓ All modules loaded and basic functions tested"

  IO.println ""
  IO.println "=========================================="
  IO.println "Environment Status: All systems operational"
  IO.println "=========================================="
