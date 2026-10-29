"""
data_loader.py
--------------
Loads the UCI Individual Household Electric Power Consumption dataset.

Dataset source:
    https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

The file `household_power_consumption.txt` is semicolon-delimited with
1-minute readings from 2006-12-16 to 2010-11-26.

Columns
-------
Date              : date in dd/mm/yyyy
Time              : time in hh:mm:ss
Global_active_power   : global minute-averaged active power (kilowatt)
Global_reactive_power : global minute-averaged reactive power (kilowatt)
Voltage               : minute-averaged voltage (volt)
Global_intensity      : global minute-averaged current intensity (ampere)
Sub_metering_1        : energy sub-metering for kitchen area (watt-hour)
Sub_metering_2        : energy sub-metering for laundry room (watt-hour)
Sub_metering_3        : energy sub-metering for electric water-heater and AC (watt-hour)

Missing values are encoded as '?' in the original file.
"""

import os
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DATASET_FILENAME = "household_power_consumption.txt"
NUMERIC_COLS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the UCI dataset file is not present."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        super().__init__(
            f"\n\n{'='*60}\n"
            f"  Dataset not found in: {data_dir}\n"
            f"{'='*60}\n"
            f"  Please download the dataset from:\n"
            f"  https://archive.ics.uci.edu/dataset/235/\n"
            f"  individual+household+electric+power+consumption\n\n"
            f"  Extract `household_power_consumption.txt` and\n"
            f"  place it in the `data/` folder, then rerun.\n"
            f"\n"
            f"  Alternatively, use Demo Mode (synthetic data)\n"
            f"  by setting demo=True when loading data.\n"
            f"{'='*60}\n"
        )


# --------------------------------------------------------------------------- #
# Real dataset loader
# --------------------------------------------------------------------------- #
def load_dataset(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the UCI household power consumption dataset from disk.

    Parameters
    ----------
    data_dir : str
        Directory containing `household_power_consumption.txt`.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame indexed by a UTC-naive DatetimeIndex.
        Numeric columns are float64; rows with ALL NaN power values
        are dropped; rows with partial NaN are kept (handled downstream).

    Raises
    ------
    DatasetNotFoundError
        If the dataset file is not present in `data_dir`.
    """
    filepath = Path(data_dir) / DATASET_FILENAME
    if not filepath.exists():
        raise DatasetNotFoundError(str(data_dir))

    logger.info("Loading dataset from %s …", filepath)

    df = pd.read_csv(
        filepath,
        sep=";",
        na_values=["?", ""],  # '?' is used for missing values
        low_memory=False,
    )

    # ---- Combine Date + Time into a single datetime index ----- #
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S"
    )
    df = df.drop(columns=["Date", "Time"])
    df = df.set_index("datetime").sort_index()

    # ---- Coerce numeric columns (safety net) ------------------- #
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where *all* numeric columns are NaN (completely missing minutes)
    df = df.dropna(how="all", subset=NUMERIC_COLS)

    logger.info(
        "Dataset loaded: %d rows from %s to %s",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df


def get_dataset_info(data_dir: str = "data") -> dict:
    """
    Return metadata about the dataset file without loading it fully.

    Returns a dict with keys: exists, filepath, size_mb, estimated_rows.
    """
    filepath = Path(data_dir) / DATASET_FILENAME
    info = {"exists": filepath.exists(), "filepath": str(filepath)}
    if filepath.exists():
        size_bytes = filepath.stat().st_size
        info["size_mb"] = round(size_bytes / (1024 ** 2), 1)
        # ~75 bytes per row on average
        info["estimated_rows"] = int(size_bytes / 75)
    return info


# --------------------------------------------------------------------------- #
# Synthetic data generator (Demo Mode)
# --------------------------------------------------------------------------- #
def generate_synthetic_data(
    n_days: int = 90,
    freq: str = "1min",
    seed: int = 42,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
    """
    Generate realistic-looking synthetic household electricity data for demo mode.

    The synthetic data mimics real patterns:
    - Low consumption at night (00:00 – 06:00)
    - Moderate morning activity (06:00 – 09:00)
    - Daytime baseline with occasional spikes (09:00 – 17:00)
    - Evening peak (17:00 – 22:00)
    - Seasonal noise via a sinusoidal component

    **IMPORTANT**: This is clearly labelled synthetic/simulated data.
    It does NOT represent a real household.

    Parameters
    ----------
    n_days : int
        Number of days of synthetic data to generate.
    freq : str
        Pandas frequency string (default '1min' for 1-minute intervals).
    seed : int
        Random seed for reproducibility.
    start_date : str
        Start date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same schema as the real dataset.
    """
    rng = np.random.default_rng(seed)

    index = pd.date_range(start=start_date, periods=n_days * 24 * 60, freq=freq)
    n = len(index)
    hours = index.hour + index.minute / 60.0

    # ---- Time-of-day power profile (kilowatt) ------------------- #
    # Base profile: night low, morning ramp, daytime moderate, evening peak
    base = _hourly_profile(hours)

    # Weekly pattern: slightly lower on weekends
    weekend_mask = index.dayofweek >= 5
    base = np.where(weekend_mask, base * 0.85, base)

    # Seasonal variation (longer days in summer → less heating)
    day_of_year = index.dayofyear
    seasonal = 0.3 * np.sin(2 * np.pi * (day_of_year - 15) / 365)
    base = base - seasonal  # summer: lower heating load

    # Gaussian noise
    noise = rng.normal(0, 0.15, n)
    global_active = np.clip(base + noise, 0.1, 8.0)

    # ---- Derived columns ---------------------------------------- #
    global_reactive = global_active * rng.uniform(0.1, 0.4, n)
    voltage = rng.normal(240, 3, n)
    global_intensity = global_active * 1000 / (voltage * np.sqrt(3))

    # Sub-metering (watt-hour per minute ≈ watts):
    # SM1: kitchen (active during meals)
    sm1 = _meal_profile(hours) * rng.uniform(0.5, 1.5, n)
    sm1 = np.clip(sm1 + rng.normal(0, 2, n), 0, None)

    # SM2: laundry (occasional spikes)
    sm2_base = np.zeros(n)
    laundry_events = rng.integers(0, n, size=n_days * 2)  # ~2 per day
    for idx in laundry_events:
        duration = rng.integers(30, 90)
        end = min(idx + duration, n)
        sm2_base[idx:end] = rng.uniform(20, 50)
    sm2 = np.clip(sm2_base + rng.normal(0, 1, n), 0, None)

    # SM3: water heater/AC (morning and evening)
    sm3 = _heater_profile(hours) * rng.uniform(0.5, 1.5, n)
    sm3 = np.clip(sm3 + rng.normal(0, 1, n), 0, None)

    # ---- Inject a handful of obvious anomalies for demo interest - #
    # (these are CLEARLY SYNTHETIC — labelled in evaluation module)
    n_anomaly_blocks = max(3, n_days // 10)
    anomaly_starts = rng.integers(n // 10, 9 * n // 10, size=n_anomaly_blocks)
    for start in anomaly_starts:
        duration = rng.integers(15, 60)
        end = min(start + duration, n)
        multiplier = rng.uniform(2.5, 4.0)
        global_active[start:end] *= multiplier
        sm1[start:end] *= multiplier

    df = pd.DataFrame(
        {
            "Global_active_power": global_active.round(3),
            "Global_reactive_power": global_reactive.round(3),
            "Voltage": voltage.round(3),
            "Global_intensity": global_intensity.round(3),
            "Sub_metering_1": sm1.round(3),
            "Sub_metering_2": sm2.round(3),
            "Sub_metering_3": sm3.round(3),
        },
        index=index,
    )
    df.index.name = "datetime"

    logger.info(
        "Generated %d rows of synthetic demo data (%d days).", n, n_days
    )
    return df


# --------------------------------------------------------------------------- #
# Private helpers for synthetic profile generation
# --------------------------------------------------------------------------- #
def _hourly_profile(hours: np.ndarray) -> np.ndarray:
    """Piecewise hourly active-power profile (kilowatt) for synthetic data."""
    profile = np.zeros_like(hours)
    mask = (hours >= 0) & (hours < 6)
    profile[mask] = 0.3  # sleeping

    mask = (hours >= 6) & (hours < 9)
    profile[mask] = 0.5 + 0.8 * ((hours[mask] - 6) / 3)  # morning ramp

    mask = (hours >= 9) & (hours < 17)
    profile[mask] = 1.1  # daytime

    mask = (hours >= 17) & (hours < 22)
    profile[mask] = 2.5  # evening peak

    mask = (hours >= 22) & (hours < 24)
    profile[mask] = 0.6  # wind-down

    return profile


def _meal_profile(hours: np.ndarray) -> np.ndarray:
    """Kitchen sub-metering profile — spikes at meal times (watts)."""
    profile = np.zeros_like(hours)
    # Breakfast 7-8
    profile[(hours >= 7) & (hours < 8)] = 15.0
    # Lunch 12-13
    profile[(hours >= 12) & (hours < 13)] = 20.0
    # Dinner 18-20
    profile[(hours >= 18) & (hours < 20)] = 25.0
    return profile


def _heater_profile(hours: np.ndarray) -> np.ndarray:
    """Water heater / AC sub-metering profile (watts)."""
    profile = np.zeros_like(hours)
    profile[(hours >= 6) & (hours < 9)] = 18.0   # morning shower
    profile[(hours >= 20) & (hours < 22)] = 15.0  # evening
    return profile
