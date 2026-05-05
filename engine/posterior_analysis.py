"""
posterior_analysis.py
Extracts decision-relevant metrics from the PyMC posterior trace:
  - P(B > A): probability treatment beats control
  - HDI: 95% Highest Density Interval on lift
  - ROPE test: fraction of posterior inside a practically-equivalent zone
  - Revenue impact range: HDI × baseline GMV
"""

import numpy as np
import arviz as az
from dataclasses import dataclass


@dataclass
class PosteriorSummary:
    prob_b_beats_a: float        # P(treatment > control)
    lift_mean: float             # posterior mean of absolute lift
    lift_hdi_lower: float        # 95% HDI lower bound
    lift_hdi_upper: float        # 95% HDI upper bound
    rel_lift_mean: float         # relative lift (%)
    rope_overlap: float          # fraction of posterior inside ROPE
    control_cvr: float           # posterior mean control CVR
    treatment_cvr: float         # posterior mean treatment CVR
    revenue_low: float | None    # annual revenue impact lower bound
    revenue_high: float | None   # annual revenue impact upper bound


def analyse_posterior(
    trace: az.InferenceData,
    rope_lower: float = -0.002,
    rope_upper: float = 0.002,
    annual_baseline_gmv: float | None = None,
    hdi_prob: float = 0.95,
) -> PosteriorSummary:
    """
    Parameters
    ----------
    trace               : ArviZ InferenceData from run_bayesian_ab_test
    rope_lower/upper    : Region Of Practical Equivalence bounds (absolute lift)
    annual_baseline_gmv : Optional. If provided, computes revenue impact range.
    hdi_prob            : HDI coverage (default 95%)
    """
    posterior = trace.posterior

    lift_samples = posterior["lift"].values.flatten()
    p_control_samples = posterior["p_control"].values.flatten()
    p_treatment_samples = posterior["p_treatment"].values.flatten()
    rel_lift_samples = posterior["rel_lift"].values.flatten()

    # P(B > A)
    prob_b_beats_a = float((lift_samples > 0).mean())

    # HDI on lift
    hdi = az.hdi(trace, var_names=["lift"], hdi_prob=hdi_prob)["lift"].values
    hdi_lower, hdi_upper = float(hdi[0]), float(hdi[1])

    # ROPE overlap
    in_rope = (lift_samples >= rope_lower) & (lift_samples <= rope_upper)
    rope_overlap = float(in_rope.mean())

    # Revenue impact
    revenue_low = revenue_high = None
    if annual_baseline_gmv is not None:
        revenue_low = hdi_lower * annual_baseline_gmv
        revenue_high = hdi_upper * annual_baseline_gmv

    return PosteriorSummary(
        prob_b_beats_a=prob_b_beats_a,
        lift_mean=float(lift_samples.mean()),
        lift_hdi_lower=hdi_lower,
        lift_hdi_upper=hdi_upper,
        rel_lift_mean=float(rel_lift_samples.mean()),
        rope_overlap=rope_overlap,
        control_cvr=float(p_control_samples.mean()),
        treatment_cvr=float(p_treatment_samples.mean()),
        revenue_low=revenue_low,
        revenue_high=revenue_high,
    )
