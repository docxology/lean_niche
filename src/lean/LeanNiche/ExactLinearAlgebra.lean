import Std

/-!
Exact rational vector laws, separate from the executable Float API.
No theorem here asserts a rounding-free law about IEEE floating-point values.
-/
namespace LeanNiche.ExactLinearAlgebra

abbrev Vector (n : Nat) := Fin n → Rat
abbrev Matrix (m n : Nat) := Fin m → Fin n → Rat

def zero_vector (n : Nat) : Vector n := fun _ => 0

def vector_add {n : Nat} (v w : Vector n) : Vector n := fun i => v i + w i

def scalar_mult {n : Nat} (c : Rat) (v : Vector n) : Vector n := fun i => c * v i

theorem zero_vector_property (n : Nat) (v : Vector n) :
    vector_add v (zero_vector n) = v := by
  funext i
  simp only [vector_add, zero_vector]
  grind

theorem vector_add_comm {n : Nat} (v w : Vector n) :
    vector_add v w = vector_add w v := by
  funext i
  simp only [vector_add]
  grind

theorem vector_add_assoc {n : Nat} (u v w : Vector n) :
    vector_add (vector_add u v) w = vector_add u (vector_add v w) := by
  funext i
  simp only [vector_add]
  grind

theorem scalar_mult_distributive {n : Nat} (c : Rat) (v w : Vector n) :
    scalar_mult c (vector_add v w) = vector_add (scalar_mult c v) (scalar_mult c w) := by
  funext i
  simp only [scalar_mult, vector_add]
  grind

theorem scalar_mult_assoc {n : Nat} (a b : Rat) (v : Vector n) :
    scalar_mult a (scalar_mult b v) = scalar_mult (a * b) v := by
  funext i
  simp only [scalar_mult]
  grind

#print axioms zero_vector_property
#print axioms vector_add_comm
#print axioms vector_add_assoc
#print axioms scalar_mult_distributive
#print axioms scalar_mult_assoc

end LeanNiche.ExactLinearAlgebra
