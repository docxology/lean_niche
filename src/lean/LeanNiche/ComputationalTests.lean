import LeanNiche.Basic
import LeanNiche.LinearAlgebra
import LeanNiche.Statistics
import LeanNiche.DynamicalSystems
import LeanNiche.ControlTheory
import LeanNiche.Lyapunov

/-!
# LeanNiche Computational Tests
Real computational examples and evaluations to verify mathematical correctness.
-/

namespace LeanNiche.ComputationalTests

open LeanNiche.Basic
open LeanNiche.LinearAlgebra
open LeanNiche.Statistics
open LeanNiche.DynamicalSystems
open LeanNiche.ControlTheory
open LeanNiche.Lyapunov

/-! ## Computational Examples with #eval -/

/-- Fibonacci sequence for computational testing -/
def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

-- Test Fibonacci computations
#eval fibonacci 10  -- Should equal 55
#eval fibonacci 15  -- Should equal 610

/-- Factorial for computational testing -/
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Test factorial computations
#eval factorial 5   -- Should equal 120
#eval factorial 7   -- Should equal 5040

/-- List operations with verification -/
def list_sum (xs : List ℕ) : ℕ := xs.foldl (· + ·) 0

-- Test list operations
#eval list_sum [1, 2, 3, 4, 5]  -- Should equal 15
#eval [1, 2, 3, 4, 5].length    -- Should equal 5

/-- Real number computations -/
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Test real computations
#eval quadratic 1 2 1 3  -- Should equal 16

/-- Vector operations testing -/
def test_vector_2d : Vector 2 := fun i => if i = 0 then 3.0 else 4.0

#eval vector_norm test_vector_2d  -- Should equal 5.0

/-- Matrix operations testing -/
def test_matrix_2x2 : Matrix 2 2 := fun i j => 
  if i = 0 ∧ j = 0 then 1.0
  else if i = 0 ∧ j = 1 then 2.0  
  else if i = 1 ∧ j = 0 then 3.0
  else 4.0

#eval det_2x2 test_matrix_2x2  -- Should equal -2.0

/-- Statistical computations -/
def test_sample : Sample := {
  size := 5,
  values := [1.0, 2.0, 3.0, 4.0, 5.0]
}

#eval sample_mean test_sample      -- Should equal 3.0
#eval sample_variance test_sample  -- Should equal 2.5

/-- Dynamical systems testing -/
def test_state : State := { position := 10.0, velocity := 5.0 }

#eval (constant_velocity_system test_state).position  -- Should equal 15.0

/-- Logistic map iterations -/
def logistic_iterate (r : ℝ) (x₀ : ℝ) (n : ℕ) : ℝ :=
  Nat.iterate (logistic_map r) n x₀

#eval logistic_iterate 2.5 0.4 1   -- Should equal 0.6
#eval logistic_iterate 2.5 0.4 2   -- Should equal 0.75

/-- PID controller testing -/
def test_pid : PIDController := {
  kp := 1.0,
  ki := 0.1, 
  kd := 0.01,
  integral_state := 0.0,
  previous_error := 0.0
}

def test_pid_output := pid_control test_pid 5.0 0.1

#eval test_pid_output.1  -- Control output

/-- Lyapunov function testing -/
#eval lyapunov_candidate 3.0      -- Should equal 9.0
#eval lyapunov_candidate (-2.0)   -- Should equal 4.0

/-! ## Theorem Verification Examples -/

/-- Computational proofs that can be verified -/
theorem fibonacci_positive (n : ℕ) (h : n > 0) : fibonacci n > 0 := by
  cases n with
  | zero => contradiction
  | succ k =>
    cases k with
    | zero => simp [fibonacci]
    | succ m => 
      simp [fibonacci]
      apply Nat.add_pos_right
      apply fibonacci_positive
      simp

theorem factorial_positive (n : ℕ) : factorial n > 0 := by
  induction n with
  | zero => simp [factorial]
  | succ n ih =>
    simp [factorial]
    apply Nat.mul_pos
    · simp
    · exact ih

theorem list_sum_nonneg (xs : List ℕ) : list_sum xs ≥ 0 := by
  simp [list_sum]
  apply List.foldl_nonneg
  · rfl
  · intro a b _ _
    exact Nat.add_le_add_left (Nat.zero_le b) a

theorem quadratic_at_zero (a b c : ℝ) : quadratic a b c 0 = c := by
  simp [quadratic]

theorem vector_norm_nonneg {n : ℕ} (v : Vector n) : vector_norm v ≥ 0 := by
  simp [vector_norm]
  exact Real.sqrt_nonneg _

theorem sample_mean_in_range (s : Sample) (min_val max_val : ℝ)
  (h : ∀ x ∈ s.values, min_val ≤ x ∧ x ≤ max_val) :
  min_val ≤ sample_mean s ∧ sample_mean s ≤ max_val := by
  exact sample_mean_bounds s min_val max_val h

theorem constant_velocity_preserves_velocity (s : State) :
  (constant_velocity_system s).velocity = s.velocity := by
  simp [constant_velocity_system]

theorem logistic_map_fixed_points (r : ℝ) :
  logistic_map r 0 = 0 ∧ (r ≠ 1 → logistic_map r (1 - 1/r) = 1 - 1/r) := by
  constructor
  · simp [logistic_map]
  · intro h
    simp [logistic_map]
    field_simp
    ring

theorem pid_output_bounded (controller : PIDController) (error dt : ℝ) 
  (h_error : abs error ≤ 1) (h_dt : dt > 0) :
  let (output, _) := pid_control controller error dt
  abs output ≤ abs controller.kp + abs controller.ki + abs controller.kd := by
  exact pid_bounded_output controller error dt h_error

/-! ## Integration Tests -/

/-- Test that combines multiple modules -/
def integrated_test : ℝ :=
  let sample_data := test_sample
  let mean_val := sample_mean sample_data
  let lyap_val := lyapunov_candidate mean_val
  let state := { position := lyap_val, velocity := 1.0 : State }
  let new_state := constant_velocity_system state
  new_state.position

#eval integrated_test  -- Should equal 10.0

theorem integrated_test_correct : integrated_test = 10.0 := by
  simp [integrated_test, test_sample, sample_mean, lyapunov_candidate, constant_velocity_system]
  norm_num

/-! ## Performance and Correctness Verification -/

/-- Verify computational correctness -/
theorem computational_consistency :
  fibonacci 5 = 5 ∧ 
  factorial 4 = 24 ∧
  list_sum [1, 2, 3] = 6 := by
  simp [fibonacci, factorial, list_sum]
  norm_num

/-- Verify mathematical properties hold computationally -/
theorem mathematical_properties_verified :
  let v := test_vector_2d
  let s := test_sample
  vector_norm v = 5.0 ∧ 
  sample_mean s = 3.0 ∧
  lyapunov_candidate 0 = 0 := by
  simp [test_vector_2d, vector_norm, test_sample, sample_mean, lyapunov_candidate]
  norm_num

/-- Verify dynamical systems properties -/
theorem dynamical_systems_verified :
  let r := 2.0
  let x := 0.5
  logistic_map r x = 0.5 ∧
  is_fixed_point (logistic_map r) 0 := by
  constructor
  · simp [logistic_map]; norm_num
  · simp [is_fixed_point, logistic_map]

end LeanNiche.ComputationalTests
