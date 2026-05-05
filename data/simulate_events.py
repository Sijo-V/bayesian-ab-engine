"""
simulate_events.py
Generates a realistic A/B experiment dataset with a built-in novelty effect.
The treatment CVR is inflated in early days and decays toward the true rate,
giving the novelty detector something real to find.
"""

import numpy as np
import pandas as pd
from faker import Faker


def simulate_experiment(
    n_users: int = 10_000,
    true_control_cvr: float = 0.05,
    true_treatment_cvr: float = 0.065,
    novelty_boost: float = 0.04,
    novelty_decay_days: int = 5,
    experiment_days: int = 21,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        user_id, variant, timestamp, session_day, converted
    """
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    rows = []
    for i in range(n_users):
        variant = "control" if i % 2 == 0 else "treatment"
        day = int(rng.integers(0, experiment_days))

        if variant == "treatment":
            boost = novelty_boost * np.exp(-day / novelty_decay_days)
            cvr = min(true_treatment_cvr + boost, 1.0)
        else:
            cvr = true_control_cvr

        converted = int(rng.random() < cvr)
        rows.append(
            {
                "user_id": fake.uuid4(),
                "variant": variant,
                "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                "session_day": day,
                "converted": converted,
            }
        )

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    df = simulate_experiment()
    out = "data/experiment_events.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")
    print(df.groupby("variant")[["converted"]].agg(["count", "mean"]).round(4))
