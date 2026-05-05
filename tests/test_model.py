"""
test_model.py
Unit tests for the core engine modules.
Run with: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.simulate_events import simulate_experiment
from data.loader import load_from_dataframe
from engine.validator import validate_experiment
from engine.bayesian_model import prepare_inputs, ModelInputs
from engine.posterior_analysis import analyse_posterior
from engine.novelty_detector import detect_novelty_effect


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clean_df():
    return simulate_experiment(n_users=2000, seed=99)


@pytest.fixture(scope="module")
def model_inputs(clean_df):
    return prepare_inputs(clean_df)


# ── Data layer ────────────────────────────────────────────────────────────────

def test_simulate_returns_correct_columns(clean_df):
    assert set(clean_df.columns) >= {"user_id", "variant", "timestamp", "session_day", "converted"}


def test_simulate_balanced_variants(clean_df):
    counts = clean_df["variant"].value_counts()
    assert abs(counts["control"] - counts["treatment"]) <= 50


def test_loader_validates_missing_column():
    bad_df = pd.DataFrame({"user_id": [1], "variant": ["control"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        load_from_dataframe(bad_df)


def test_loader_validates_bad_variant():
    df = simulate_experiment(n_users=100)
    df.loc[0, "variant"] = "unknown"
    with pytest.raises(ValueError, match="Unexpected variant values"):
        load_from_dataframe(df)


# ── Validator ─────────────────────────────────────────────────────────────────

def test_validator_no_srm_on_balanced(clean_df):
    result = validate_experiment(clean_df)
    assert not result.srm_detected


def test_validator_detects_srm():
    df = simulate_experiment(n_users=2000)
    # Force 80/20 split
    df.loc[df["variant"] == "treatment", "variant"] = "control"
    df.loc[df.index[:400], "variant"] = "treatment"
    result = validate_experiment(df)
    assert result.srm_detected


def test_validator_insufficient_sample():
    df = simulate_experiment(n_users=200)
    result = validate_experiment(df, min_per_variant=500)
    assert not result.sufficient_sample


# ── Model inputs ──────────────────────────────────────────────────────────────

def test_prepare_inputs_non_negative(model_inputs):
    assert model_inputs.control_trials > 0
    assert model_inputs.treatment_trials > 0
    assert model_inputs.control_conversions >= 0
    assert model_inputs.treatment_conversions >= 0


def test_prepare_inputs_conversions_leq_trials(model_inputs):
    assert model_inputs.control_conversions <= model_inputs.control_trials
    assert model_inputs.treatment_conversions <= model_inputs.treatment_trials


# ── Novelty detector ──────────────────────────────────────────────────────────

def test_novelty_detector_returns_result(clean_df):
    result = detect_novelty_effect(clean_df)
    assert 0.0 <= result.novelty_score <= 1.0
    assert result.risk_level in ("Low", "Medium", "High", "Unknown")


def test_novelty_detector_finds_novelty_in_simulated():
    # Simulate with strong novelty boost — should detect it
    df = simulate_experiment(n_users=8000, novelty_boost=0.08, novelty_decay_days=3, seed=7)
    result = detect_novelty_effect(df)
    # With a large boost and fast decay, novelty score should be > 0
    assert result.novelty_score > 0 or result.slope < 0  # at minimum slope is negative


def test_novelty_detector_no_effect():
    # No novelty boost — should have low score
    df = simulate_experiment(n_users=8000, novelty_boost=0.0, seed=5)
    result = detect_novelty_effect(df)
    assert result.novelty_score < 0.5


def test_novelty_detector_too_few_days():
    df = simulate_experiment(n_users=500, experiment_days=3, seed=1)
    result = detect_novelty_effect(df)
    assert result.risk_level == "Unknown" or result.novelty_score == 0.0
