import LeanNiche.Basic
import LeanNiche.LinearAlgebra
import LeanNiche.Statistics
import LeanNiche.DynamicalSystems
import LeanNiche.ControlTheory
import LeanNiche.Lyapunov

/-! ## Real Lean Computational Verification -/

/-- Fibonacci sequence for computational testing -/
def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

-- REAL COMPUTATIONAL EVALUATIONS
#eval fibonacci 10  -- Should equal 55
#eval fibonacci 15  -- Should equal 610

/-- Factorial for computational testing -/
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Test factorial computations
#eval factorial 5   -- Should equal 120
#eval factorial 7   -- Should equal 5040

/-- List operations -/
def list_sum (xs : List ℕ) : ℕ := xs.foldl (· + ·) 0

#eval list_sum [1, 2, 3, 4, 5]  -- Should equal 15
#eval [1, 2, 3, 4, 5].length    -- Should equal 5

/-- Real number computations -/
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

#eval quadratic 1 2 1 3  -- Should equal 16

/-- Vector operations using LeanNiche.LinearAlgebra -/
open LeanNiche.LinearAlgebra

def test_vector : LeanNiche.LinearAlgebra.Vector 2 := fun i => if i = 0 then 3.0 else 4.0

#eval vector_norm test_vector  -- Should equal 5.0

/-- Statistical computations using LeanNiche.Statistics -/
open LeanNiche.Statistics

def test_sample : Sample := {
  size := 5,
  values := [1.0, 2.0, 3.0, 4.0, 5.0]
}

#eval sample_mean test_sample      -- Should equal 3.0
#eval sample_variance test_sample  -- Should equal 2.5

/-- Dynamical systems using LeanNiche.DynamicalSystems -/
open LeanNiche.DynamicalSystems

def test_state : State := { position := 10.0, velocity := 5.0 }

#eval (constant_velocity_system test_state).position  -- Should equal 15.0

#eval logistic_map 2.5 0.4  -- Should equal 0.6

/-- PID controller using LeanNiche.ControlTheory -/
open LeanNiche.ControlTheory

def test_pid : PIDController := {
  kp := 1.0,
  ki := 0.1, 
  kd := 0.01,
  integral_state := 0.0,
  previous_error := 0.0
}

#eval (pid_control test_pid 5.0 0.1).1  -- Control output

/-- Lyapunov functions using LeanNiche.Lyapunov -/
open LeanNiche.Lyapunov

#eval lyapunov_candidate 3.0      -- Should equal 9.0
#eval lyapunov_candidate (-2.0)   -- Should equal 4.0

/-! ## THEOREM VERIFICATION -/

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

theorem computational_correctness :
  fibonacci 5 = 5 ∧ 
  factorial 4 = 24 ∧
  list_sum [1, 2, 3] = 6 := by
  simp [fibonacci, factorial, list_sum]
  norm_num

#check computational_correctness
#print axioms computational_correctness
