"""
loader.py
Loads experiment event data from CSV, SQLite, or a pandas DataFrame.
Validates required columns and returns a clean DataFrame ready for the engine.
"""

import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = {"user_id", "variant", "timestamp", "session_day", "converted"}
VALID_VARIANTS = {"control", "treatment"}


def load_from_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return _validate(df)


def load_from_sqlite(db_path: str, table: str = "events") -> pd.DataFrame:
    import sqlite3
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table}", conn, parse_dates=["timestamp"])
    conn.close()
    return _validate(df)


def load_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _validate(df.copy())


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    invalid_variants = set(df["variant"].unique()) - VALID_VARIANTS
    if invalid_variants:
        raise ValueError(f"Unexpected variant values: {invalid_variants}. Expected: {VALID_VARIANTS}")

    if not pd.api.types.is_integer_dtype(df["converted"]):
        df["converted"] = df["converted"].astype(int)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
