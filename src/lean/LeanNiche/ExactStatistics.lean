import Std

/-! Exact statistical identities. These definitions do not model Float rounding or NaN. -/
namespace LeanNiche.ExactStatistics

structure Sample where
  size : Nat
  values : List Rat

def sample_mean (s : Sample) : Rat :=
  if s.size = 0 then 0 else s.values.sum / (s.size : Rat)

structure ConfidenceInterval where
  lower : Rat
  upper : Rat
  confidence_level : Rat

theorem sample_mean_singleton (x : Rat) : sample_mean ⟨1, [x]⟩ = x := by
  simp [sample_mean]
  grind

theorem confidence_interval_order (ci : ConfidenceInterval) :
    ci.lower ≤ ci.upper ∨ ci.lower > ci.upper := by
  grind

#print axioms sample_mean_singleton
#print axioms confidence_interval_order

end LeanNiche.ExactStatistics
