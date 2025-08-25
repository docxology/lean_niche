-- Import working LeanNiche modules only
import LeanNiche.Basic
import LeanNiche.LinearAlgebra
import LeanNiche.Statistics
import LeanNiche.Utils
import LeanNiche.Tactics
import LeanNiche.Computational
import LeanNiche.WorkingComputations

/-!
# LeanNiche Root Module
Main entry point for the LeanNiche mathematical library.
-/

/-!
## LeanNiche Library

This is the main LeanNiche library providing:
- Basic mathematical foundations
- Linear algebra operations (basic)
- Statistical analysis tools
- Computational verification tools
- Proof tactics and utilities
-/

namespace LeanNiche

-- Re-export key definitions for easy access
open LeanNiche.Basic
open LeanNiche.LinearAlgebra
open LeanNiche.Statistics
open LeanNiche.Computational
open LeanNiche.WorkingComputations

-- Library version information
def version : String := "0.1.0"

-- Main library verification
theorem library_loaded : True := trivial

end LeanNiche