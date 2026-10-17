import LeanNiche.Basic

/-!
# LeanNiche Linear Algebra Module
Basic linear algebra operations for control theory and dynamical systems.
-/

namespace LeanNiche.LinearAlgebra

open LeanNiche.Basic

-- Vector type as function from Fin n to Float
def Vector (n : Nat) : Type := Fin n → Float

-- Matrix type as function from Fin m × Fin n to Float
def Matrix (m n : Nat) : Type := Fin m → Fin n → Float

-- Zero vector
def zero_vector (n : Nat) : Vector n := fun _ => 0.0

-- Zero matrix
def zero_matrix (m n : Nat) : Matrix m n := fun _ _ => 0.0

-- Identity matrix
def identity_matrix (n : Nat) : Matrix n n := 
  fun i j => if i = j then 1.0 else 0.0

-- Vector addition
def vector_add {n : Nat} (v w : Vector n) : Vector n :=
  fun i => v i + w i

-- Scalar multiplication
def scalar_mult {n : Nat} (c : Float) (v : Vector n) : Vector n :=
  fun i => c * v i

-- Basic linear algebra theorems

-- Identity matrix properties
theorem identity_matrix_diag (n : Nat) (i : Fin n) :
  identity_matrix n i i = 1.0 := by
  simp [identity_matrix]

theorem identity_matrix_off_diag (n : Nat) (i j : Fin n) (h : i ≠ j) :
  identity_matrix n i j = 0.0 := by
  simp [identity_matrix, h]

end LeanNiche.LinearAlgebra