/-!
Executable regression witnesses for legacy Float algebra claims.
These are observed floating-point computations, not theorem certificates.
-/
#eval let a : Float := 10000000000000000.0
      let b : Float := -10000000000000000.0
      let c : Float := 1.0
      (((a + b) + c), (a + (b + c)), ((a + b) + c == a + (b + c)))
#eval let a : Float := 2.0
      let b : Float := -2.0
      let c : Float := 1.0e308
      (c * (a + b), c * a + c * b, c * (a + b) == c * a + c * b)
#eval let nan : Float := 0.0 / 0.0
      (decide (nan ≤ 1.0), decide (nan > 1.0), decide (nan ≤ 1.0) || decide (nan > 1.0))
