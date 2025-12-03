from llm_utils import verify_proof


# Use this to manually verify a proof

print(verify_proof(
    """
      theorem algebra_sqineq_unitcircatbpabsamblt1
      (a b: ℝ)
      (h₀ : a^2 + b^2 = 1) :
      a * b + |a - b| ≤ 1 :=
        have h_main : a * b + |a - b| ≤ 1 := by
          cases' le_total 0 (a - b) with h h <;>
          simp_all [abs_of_nonneg, abs_of_nonpos, sub_nonneg, sub_nonpos] <;>
          (try
            {
              nlinarith [sq_nonneg (a - b), sq_nonneg (a + b), sq_nonneg (a + b - 1),
                sq_nonneg (a + b + 1), sq_nonneg (a - b - 1), sq_nonneg (a - b + 1)]
            }) <;>
          (try
            {
              nlinarith [sq_nonneg (a - b), sq_nonneg (a + b), sq_nonneg (a + b - 1),
                sq_nonneg (a + b + 1), sq_nonneg (a - b - 1), sq_nonneg (a - b + 1),
                sq_nonneg (a - 1), sq_nonneg (b - 1), sq_nonneg (a + 1), sq_nonneg (b + 1)]
            }) <;>
          (try
            {
              nlinarith [sq_nonneg (a - b), sq_nonneg (a + b), sq_nonneg (a + b - 1),
                sq_nonneg (a + b + 1), sq_nonneg (a - b - 1), sq_nonneg (a - b + 1),
                sq_nonneg (a - 1), sq_nonneg (b - 1), sq_nonneg (a + 1), sq_nonneg (b + 1),
                mul_self_nonneg (a - b), mul_self_nonneg (a + b)]
            })
          <;>
          (try
            {
              cases' le_total 0 (a - 1) with h₁ h₁ <;>
              cases' le_total 0 (b - 1) with h₂ h₂ <;>
              nlinarith [sq_nonneg (a - b), sq_nonneg (a + b), sq_nonneg (a + b - 1),
                sq_nonneg (a + b + 1), sq_nonneg (a - b - 1), sq_nonneg (a - b + 1),
                sq_nonneg (a - 1), sq_nonneg (b - 1), sq_nonneg (a + 1), sq_nonneg (b + 1)]
            })
        exact h_main
      """
))