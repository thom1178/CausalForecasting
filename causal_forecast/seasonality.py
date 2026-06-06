import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Tuple

from .typing import VariableType
from .utils import (
    build_label_encoder,
    encode_values,
    build_one_hot_encoder,
)

SeasonalityType = Literal["weekly", "monthly", "yearly"]

FREQUENCY_PERIOD_MAP: Dict[str, Dict[str, int]] = {
    "daily": {"weekly": 7, "monthly": 30, "yearly": 365},
    "weekly": {"monthly": 4, "yearly": 52},
    "monthly": {"yearly": 12},
    "yearly": {},
}


def infer_data_frequency(time_delta: timedelta) -> str:
    """Map a median timestep to a coarse data frequency label."""
    days = time_delta.total_seconds() / 86400.0

    if days <= 2:
        return "daily"
    if days <= 10:
        return "weekly"
    if days <= 45:
        return "monthly"
    return "yearly"


def seasonal_periods(
    frequency: str,
    seasonality_names: List[SeasonalityType],
) -> Dict[str, int]:
    """Return lag step counts for each requested seasonality at the given frequency."""
    available = FREQUENCY_PERIOD_MAP.get(frequency, {})
    periods: Dict[str, int] = {}

    for name in seasonality_names:
        if name not in available:
            raise ValueError(
                f"Seasonality '{name}' is not applicable for {frequency} data. "
                f"Available: {list(available.keys())}"
            )
        periods[name] = available[name]

    return periods


def validate_seasonality(
    data_len: int,
    lookback: int,
    requested: List[SeasonalityType],
    frequency: str,
) -> Dict[str, int]:
    """
    Enable seasonal lags only when enough history exists.

    Raises ValueError if a requested seasonality cannot be satisfied.
    """
    if not requested:
        return {}

    candidate = seasonal_periods(frequency, requested)
    enabled: Dict[str, int] = {}

    for name, period in candidate.items():
        min_rows = period + lookback + 1
        if data_len >= min_rows:
            enabled[name] = period
        else:
            raise ValueError(
                f"Not enough data for '{name}' seasonality (period={period}). "
                f"Need at least {min_rows} rows, got {data_len}."
            )

    return enabled


def _cyclical_pair(values: pd.Series, period: float, prefix: str) -> Tuple[pd.Series, pd.Series]:
    sin_col = np.sin(2 * math.pi * values / period)
    cos_col = np.cos(2 * math.pi * values / period)
    return sin_col, cos_col


def add_cyclical_time_features(
    df: pd.DataFrame,
    time_column: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add sin/cos cyclical encodings for weekly and yearly seasonal patterns."""
    df = df.copy()
    dt = pd.to_datetime(df[time_column])

    dow_sin, dow_cos = _cyclical_pair(dt.dt.dayofweek.astype(float), 7, "dayofweek")
    df["dayofweek_sin"] = dow_sin
    df["dayofweek_cos"] = dow_cos

    month_sin, month_cos = _cyclical_pair(dt.dt.month.astype(float), 12, "month")
    df["month_sin"] = month_sin
    df["month_cos"] = month_cos

    dayofyear = dt.dt.dayofyear.astype(float)
    doy_sin, doy_cos = _cyclical_pair(dayofyear, 365.25, "dayofyear")
    df["dayofyear_sin"] = doy_sin
    df["dayofyear_cos"] = doy_cos

    feature_cols = [
        "dayofweek_sin",
        "dayofweek_cos",
        "month_sin",
        "month_cos",
        "dayofyear_sin",
        "dayofyear_cos",
    ]
    return df, feature_cols


def cyclical_features_from_date(dt: datetime) -> Dict[str, float]:
    """Compute cyclical time features for a single future timestamp."""
    dow = float(dt.weekday())
    month = float(dt.month)
    dayofyear = float(dt.timetuple().tm_yday)

    return {
        "dayofweek_sin": math.sin(2 * math.pi * dow / 7),
        "dayofweek_cos": math.cos(2 * math.pi * dow / 7),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "dayofyear_sin": math.sin(2 * math.pi * dayofyear / 365.25),
        "dayofyear_cos": math.cos(2 * math.pi * dayofyear / 365.25),
    }


def decomposition_period(
    seasonality: SeasonalityType = "weekly",
    time_delta: Optional[timedelta] = None,
) -> int:
    """Infer statsmodels decomposition period from seasonality type and data frequency."""
    if time_delta is None:
        time_delta = timedelta(days=1)

    frequency = infer_data_frequency(time_delta)
    periods = FREQUENCY_PERIOD_MAP.get(frequency, {})

    if seasonality in periods:
        return periods[seasonality]

    defaults = {"weekly": 7, "monthly": 30, "yearly": 365}
    return defaults.get(seasonality, 7)


def add_seasonal_lag_features(
    df: pd.DataFrame,
    var: str,
    seasonal_periods_map: Dict[str, int],
    variable_type: VariableType,
    label_encoders: dict,
    one_hot_encoders: dict,
    one_hot_feature_names: dict,
    use_one_hot: bool,
) -> List[str]:
    """Add seasonal lag features for one variable."""
    feature_cols: List[str] = []

    for season_name, period in seasonal_periods_map.items():
        col = f"{var}_s_lag_{season_name}"

        if variable_type == "continuous":
            df[col] = df[var].shift(period)
            feature_cols.append(col)
        elif use_one_hot and variable_type in ("multiclass", "ordinal"):
            if var not in one_hot_encoders:
                encoder, names = build_one_hot_encoder(df[var])
                one_hot_encoders[var] = encoder
                one_hot_feature_names[var] = names

            encoder = one_hot_encoders[var]
            names = one_hot_feature_names[var]

            lagged = df[var].shift(period)
            missing = lagged.isna()
            lagged_str = lagged.astype(str).values.reshape(-1, 1)
            encoded = encoder.transform(lagged_str)

            for idx, name in enumerate(names):
                sub_col = f"{col}_{name}"
                df[sub_col] = encoded[:, idx]
                df.loc[missing, sub_col] = np.nan
                feature_cols.append(sub_col)
        else:
            if var not in label_encoders:
                label_encoders[var] = build_label_encoder(df[var], variable_type)

            encoded = encode_values(df[var], label_encoders[var])
            df[f"{var}_encoded"] = encoded
            df[col] = df[f"{var}_encoded"].shift(period)
            feature_cols.append(col)

    return feature_cols
