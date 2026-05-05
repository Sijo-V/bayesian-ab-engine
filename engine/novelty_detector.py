"""
novelty_detector.py
Detects novelty effects in A/B experiments.

A novelty effect occurs when users interact with a new treatment simply
because it is new — producing an inflated early lift that decays over time.
This module detects it by:
  1. Computing daily rolling conversion rates per variant
  2. Calculating daily lift (treatment CVR − control CVR) over time
  3. Fitting a linear regression on the daily lift series
  4. Returning a novelty_score (0–1) based on slope direction and significance
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field


@dataclass
class NoveltyResult:
    novelty_score: float          # 0 = none, 1 = strong novelty effect
    risk_level: str               # "Low" / "Medium" / "High"
    slope: float                  # lift change per day (negative = decaying)
    slope_p_value: float          # significance of the slope
    daily_lifts: list[dict]       # list of {day, lift, n_control, n_treatment}
    recommendation: str


def detect_novelty_effect(
    df: pd.DataFrame,
    window_days: int = 1,
    min_users_per_window: int = 150,
    significance_threshold: float = 0.10,
) -> NoveltyResult:
    """
    Parameters
    ----------
    df                     : experiment DataFrame (variant, session_day, converted)
    window_days            : days per rolling window (1 = daily)
    min_users_per_window   : skip windows with fewer users than this
    significance_threshold : p-value threshold for slope significance

    Returns
    -------
    NoveltyResult with score, risk level, slope stats, and daily lift series
    """
    days = sorted(df["session_day"].unique())
    daily_lifts = []

    for day in days:
        window = df[df["session_day"].between(day, day + window_days - 1)]
        ctrl = window[window["variant"] == "control"]
        trt = window[window["variant"] == "treatment"]

        if len(window) < min_users_per_window or len(ctrl) == 0 or len(trt) == 0:
            continue

        lift = trt["converted"].mean() - ctrl["converted"].mean()
        daily_lifts.append(
            {
                "day": int(day),
                "lift": float(lift),
                "control_cvr": float(ctrl["converted"].mean()),
                "treatment_cvr": float(trt["converted"].mean()),
                "n_control": len(ctrl),
                "n_treatment": len(trt),
            }
        )

    if len(daily_lifts) < 4:
        return NoveltyResult(
            novelty_score=0.0,
            risk_level="Unknown",
            slope=0.0,
            slope_p_value=1.0,
            daily_lifts=daily_lifts,
            recommendation="Not enough daily data points to assess novelty effect (need 4+).",
        )

    x = np.array([d["day"] for d in daily_lifts])
    y = np.array([d["lift"] for d in daily_lifts])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    # Score: high if slope is negative AND statistically significant
    if p_value < significance_threshold and slope < 0:
        # Scale: slope of -0.002/day over 21 days = substantial decay
        novelty_score = min(abs(slope) * 200, 1.0)
    else:
        novelty_score = 0.0

    if novelty_score < 0.3:
        risk_level = "Low"
        recommendation = "No significant novelty effect detected. Safe to make a go/no-go decision."
    elif novelty_score < 0.6:
        risk_level = "Medium"
        recommendation = (
            "Marginal novelty effect detected. Consider extending the experiment "
            "by 7 days and focusing on users who enrolled after day 7."
        )
    else:
        risk_level = "High"
        recommendation = (
            "Strong novelty effect detected. Lift is likely overstated. "
            "Segment by enrollment cohort and re-evaluate after day 14."
        )

    return NoveltyResult(
        novelty_score=round(novelty_score, 3),
        risk_level=risk_level,
        slope=round(float(slope), 6),
        slope_p_value=round(float(p_value), 4),
        daily_lifts=daily_lifts,
        recommendation=recommendation,
    )
