"""
bayesian_model.py
Bayesian A/B test using a Beta-Binomial conjugate model in PyMC.
Samples the posterior over conversion rates for control and treatment,
then derives lift and relative lift as deterministic quantities.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from dataclasses import dataclass


@dataclass
class ModelInputs:
    control_trials: int
    control_conversions: int
    treatment_trials: int
    treatment_conversions: int


def prepare_inputs(df: pd.DataFrame) -> ModelInputs:
    ctrl = df[df["variant"] == "control"]
    trt = df[df["variant"] == "treatment"]
    return ModelInputs(
        control_trials=len(ctrl),
        control_conversions=int(ctrl["converted"].sum()),
        treatment_trials=len(trt),
        treatment_conversions=int(trt["converted"].sum()),
    )


def run_bayesian_ab_test(
    inputs: ModelInputs,
    draws: int = 2000,
    tune: int = 1000,
    chains: int = 4,
    target_accept: float = 0.9,
    random_seed: int = 42,
) -> az.InferenceData:
    """
    Fits a Beta-Binomial model and returns an ArviZ InferenceData object.

    Parameters
    ----------
    inputs       : ModelInputs with trial/conversion counts per variant
    draws        : posterior draws per chain
    tune         : warmup steps
    chains       : number of MCMC chains
    target_accept: NUTS target acceptance rate (0.8–0.95)
    random_seed  : reproducibility

    Returns
    -------
    ArviZ InferenceData with variables:
        p_control, p_treatment, lift, rel_lift
    """
    with pm.Model() as model:
        # Weakly informative Beta(1,1) priors = uniform over [0,1]
        p_control = pm.Beta("p_control", alpha=1, beta=1)
        p_treatment = pm.Beta("p_treatment", alpha=1, beta=1)

        # Likelihoods (observed conversion counts)
        pm.Binomial(
            "obs_control",
            n=inputs.control_trials,
            p=p_control,
            observed=inputs.control_conversions,
        )
        pm.Binomial(
            "obs_treatment",
            n=inputs.treatment_trials,
            p=p_treatment,
            observed=inputs.treatment_conversions,
        )

        # Derived quantities of interest
        lift = pm.Deterministic("lift", p_treatment - p_control)
        pm.Deterministic("rel_lift", lift / p_control)

        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            random_seed=random_seed,
            progressbar=True,
            return_inferencedata=True,
        )

    return trace
