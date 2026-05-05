"""
validator.py
Checks experiment health before running the Bayesian model.
  - Sample Ratio Mismatch (SRM) via chi-squared test
  - Minimum sample size check
"""

import pandas as pd
from scipy import stats
from dataclasses import dataclass


@dataclass
class ValidationResult:
    srm_detected: bool
    srm_p_value: float
    control_n: int
    treatment_n: int
    sufficient_sample: bool
    warnings: list[str]


def validate_experiment(df: pd.DataFrame, min_per_variant: int = 500) -> ValidationResult:
    counts = df["variant"].value_counts()
    control_n = int(counts.get("control", 0))
    treatment_n = int(counts.get("treatment", 0))
    total = control_n + treatment_n

    # SRM: expected 50/50 split
    chi2, p_value = stats.chisquare([control_n, treatment_n], f_exp=[total / 2, total / 2])
    srm_detected = p_value < 0.01

    warnings = []
    if srm_detected:
        warnings.append(
            f"Sample Ratio Mismatch detected (p={p_value:.4f}). "
            f"Control={control_n}, Treatment={treatment_n}. "
            "Check assignment logic before trusting results."
        )

    sufficient = control_n >= min_per_variant and treatment_n >= min_per_variant
    if not sufficient:
        warnings.append(
            f"Insufficient sample. Need {min_per_variant} per variant, "
            f"got control={control_n}, treatment={treatment_n}."
        )

    return ValidationResult(
        srm_detected=srm_detected,
        srm_p_value=float(p_value),
        control_n=control_n,
        treatment_n=treatment_n,
        sufficient_sample=sufficient,
        warnings=warnings,
    )
