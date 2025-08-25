/-!
# LeanNiche Main Module
Main entry point and exports for LeanNiche.
-/

namespace LeanNiche

/-- Main LeanNiche theorem -/
theorem main_identity (x : Nat) : x + 0 = x := by
  induction x with
  | zero => rfl
  | succ n ih => rw [Nat.succ_add, ih]

end LeanNiche
