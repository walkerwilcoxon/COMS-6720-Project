import Mathlib
import Aesop

set_option linter.all false
set_option maxHeartbeats 0

open scoped BigOperators
open Real Nat Topology Rat Finset

theorem mathd_algebra_478
  (b h v : ℝ)
  (h₀ : 0 < b ∧ 0 < h ∧ 0 < v)
  (h₁ : v = 1 / 3 * (b * h))
  (h₂ : b = 30)
  (h₃ : h = 13 / 2) :
  v = 65 :=
  sorry

theorem numbertheory_4x3m7y3neq2003
  (x y : ℤ) :
  4 * x^3 - 7 * y^3 ≠ 2003 :=
  sorry

theorem aime_1983_p1
  (x y z w : ℕ)
  (ht : 1 < x ∧ 1 < y ∧ 1 < z)
  (hw : 0 ≤ w)
  (h0 : Real.log w / Real.log x = 24)
  (h1 : Real.log w / Real.log y = 40)
  (h2 : Real.log w / Real.log (x * y * z) = 12):
  Real.log w / Real.log z = 60 :=
  sorry