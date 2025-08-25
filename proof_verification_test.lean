/-! ## Proof Verification Test - Clean Version -/

/-- Fibonacci sequence -/
def fibonacci : Nat → Nat
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

/-- Factorial function -/
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- THEOREM 1: Fibonacci positivity -/
theorem fibonacci_positive (n : Nat) (h : n > 0) : fibonacci n > 0 := by
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

/-- THEOREM 2: Factorial positivity -/
theorem factorial_positive (n : Nat) : factorial n > 0 := by
  induction n with
  | zero => simp [factorial]
  | succ n ih =>
    simp [factorial]
    exact Nat.mul_pos (Nat.succ_pos n) ih

/-- THEOREM 3: Computational correctness -/
theorem computational_correctness :
  fibonacci 5 = 5 ∧ 
  factorial 4 = 24 := by
  simp [fibonacci, factorial]

/-- THEOREM 4: Fibonacci monotonicity -/
theorem fibonacci_monotone (n : Nat) : fibonacci n ≤ fibonacci (n + 1) := by
  cases n with
  | zero => simp [fibonacci]
  | succ m =>
    cases m with
    | zero => simp [fibonacci]
    | succ k => 
      simp [fibonacci]
      exact Nat.le_add_left _ _

/-- THEOREM 5: Basic arithmetic -/
theorem basic_arithmetic : 2 + 3 = 5 := by simp

/-- THEOREM 6: Fibonacci base cases -/
theorem fibonacci_base_cases : fibonacci 0 = 0 ∧ fibonacci 1 = 1 := by
  simp [fibonacci]

/-- THEOREM 7: Factorial base case -/
theorem factorial_base_case : factorial 0 = 1 := by
  simp [factorial]

-- Check which theorems compiled successfully
section ProofVerification

#check fibonacci_positive
#check factorial_positive  
#check computational_correctness
#check fibonacci_monotone
#check basic_arithmetic
#check fibonacci_base_cases
#check factorial_base_case

-- Check axiom dependencies
#print axioms fibonacci_positive
#print axioms factorial_positive
#print axioms computational_correctness
#print axioms fibonacci_monotone
#print axioms basic_arithmetic
#print axioms fibonacci_base_cases
#print axioms factorial_base_case

end ProofVerification
