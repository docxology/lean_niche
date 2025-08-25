/-!
# LeanNiche Set Theory Module
Basic set theory foundations for LeanNiche.
-/

namespace LeanNiche

/-- Simple set membership -/
inductive SetMembership (x : Nat) : Prop where
  | member : SetMembership x

/-- Set theory theorem -/
theorem set_membership_refl (x : Nat) : SetMembership x := by
  exact SetMembership.member

end LeanNiche
