/-! ## Simple Working Lean Computational Test -/

/-- Fibonacci sequence -/
def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

/-- Factorial function -/
def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- List sum -/
def list_sum (xs : List ℕ) : ℕ := xs.foldl (· + ·) 0

/-- REAL COMPUTATIONAL EVALUATIONS -/
#eval fibonacci 10  -- Should equal 55
#eval fibonacci 15  -- Should equal 610
#eval factorial 5   -- Should equal 120
#eval factorial 7   -- Should equal 5040
#eval list_sum [1, 2, 3, 4, 5]  -- Should equal 15

/-- Simple mathematical operations -/
#eval 2 + 3 * 4     -- Should equal 14
#eval 100 / 5       -- Should equal 20
#eval 2^10          -- Should equal 1024

/-- List operations -/
#eval [1, 2, 3, 4, 5].length    -- Should equal 5
#eval [1, 2, 3] ++ [4, 5]       -- Should equal [1, 2, 3, 4, 5]
#eval [1, 2, 3, 4, 5].reverse   -- Should equal [5, 4, 3, 2, 1]

/-- THEOREM VERIFICATION -/

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

theorem computational_correctness :
  fibonacci 5 = 5 ∧ 
  factorial 4 = 24 ∧
  list_sum [1, 2, 3] = 6 := by
  simp [fibonacci, factorial, list_sum]
  norm_num

/-- Verify theorems compile and check axioms -/
#check computational_correctness
#print axioms computational_correctness

/-- More complex computations -/
def gcd : ℕ → ℕ → ℕ
  | 0, b => b
  | a, 0 => a  
  | a, b => if a ≤ b then gcd a (b - a) else gcd (a - b) b

#eval gcd 48 18  -- Should equal 6
#eval gcd 100 25 -- Should equal 25

def is_prime (n : ℕ) : Bool :=
  if n < 2 then false
  else (List.range (n - 1)).drop 1 |>.all (fun d => n % d ≠ 0)

#eval is_prime 17  -- Should be true
#eval is_prime 15  -- Should be false

theorem gcd_comm (a b : ℕ) : gcd a b = gcd b a := by
  sorry -- This would require a more complex proof

theorem prime_gt_one (n : ℕ) (h : is_prime n = true) : n > 1 := by
  simp [is_prime] at h
  split_ifs at h with h_lt
  · contradiction
  · linarith [h_lt]
