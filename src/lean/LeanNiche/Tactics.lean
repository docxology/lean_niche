/-!
# LeanNiche Tactics Module
Custom tactics and proof automation for LeanNiche.
-/

namespace LeanNiche

/-- Simple tactic for basic arithmetic -/
macro "basic_arith" : tactic => `(tactic| rfl)

/-- Tactics theorem -/
theorem tactics_demo : 1 + 1 = 2 := by
  basic_arith

end LeanNiche
