import networkx as nx
import numpy as np
import pandas as pd
import pytest
from datetime import timedelta

from causal_forecast import CausalForecaster
from causal_forecast.seasonality import (
    add_cyclical_time_features,
    decomposition_period,
    infer_data_frequency,
    seasonal_periods,
    validate_seasonality,
)
from causal_forecast.utils import infer_time_delta


def test_infer_data_frequency_daily_weekly_monthly():
    assert infer_data_frequency(timedelta(days=1)) == "daily"
    assert infer_data_frequency(timedelta(days=7)) == "weekly"
    assert infer_data_frequency(timedelta(days=30)) == "monthly"


def test_seasonal_periods_daily():
    periods = seasonal_periods("daily", ["weekly", "monthly", "yearly"])
    assert periods == {"weekly": 7, "monthly": 30, "yearly": 365}


def test_seasonal_periods_weekly():
    periods = seasonal_periods("weekly", ["monthly", "yearly"])
    assert periods == {"monthly": 4, "yearly": 52}


def test_validate_seasonality_raises_on_insufficient_data():
    with pytest.raises(ValueError, match="Not enough data"):
        validate_seasonality(100, lookback=3, requested=["yearly"], frequency="daily")


def test_backward_compat_no_seasonality_features():
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "x": np.random.normal(0, 1, len(dates)),
            "y": np.random.normal(0, 1, len(dates)),
        }
    )
    graph = nx.DiGraph()
    graph.add_edge("x", "y")

    forecaster = CausalForecaster(data, graph, "y", "timestamp", lookback_periods=3)
    forecaster.fit(verbose=False)
    X, _ = forecaster._prepare_time_series_data("y")

    assert not any("_s_lag_" in col for col in X.columns)
    assert "dayofweek_sin" not in X.columns


def test_seasonal_lag_columns_when_enabled():
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "x": np.random.normal(0, 1, len(dates)),
            "y": np.random.normal(0, 1, len(dates)),
        }
    )
    graph = nx.DiGraph()
    graph.add_edge("x", "y")

    forecaster = CausalForecaster(
        data,
        graph,
        "y",
        "timestamp",
        lookback_periods=3,
        seasonality=["weekly", "monthly", "yearly"],
    )
    forecaster.fit(verbose=False)
    X, _ = forecaster._prepare_time_series_data("y")

    assert "x_s_lag_weekly" in X.columns
    assert "x_s_lag_monthly" in X.columns
    assert "x_s_lag_yearly" in X.columns
    assert "dayofweek_sin" in X.columns
    assert forecaster.max_history == 365


def test_predict_with_seasonality_no_column_mismatch():
    dates = pd.date_range("2023-01-01", periods=400, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "x": np.random.normal(0, 1, len(dates)),
            "y": np.random.normal(0, 1, len(dates)),
        }
    )
    graph = nx.DiGraph()
    graph.add_edge("x", "y")

    forecaster = CausalForecaster(
        data,
        graph,
        "y",
        "timestamp",
        lookback_periods=3,
        seasonality=["weekly"],
    )
    forecaster.fit(verbose=False)
    pred = forecaster.predict(steps=5)
    assert len(pred) == 5


def test_seasonality_improves_weekly_pattern_in_sample():
    n = 730
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    signal = 20 * np.sin(2 * np.pi * t / 7)
    noise = np.random.default_rng(42).normal(0, 0.1, n)
    values = signal + noise

    data = pd.DataFrame({"timestamp": dates, "x": values})
    graph = nx.DiGraph()
    graph.add_node("x")

    baseline = CausalForecaster(data, graph, "x", "timestamp", lookback_periods=3)
    baseline.fit(verbose=False)
    baseline_rmse = baseline.evaluate(in_sample=True).loc["x", "rmse"]

    seasonal = CausalForecaster(
        data,
        graph,
        "x",
        "timestamp",
        lookback_periods=3,
        seasonality=["weekly"],
    )
    seasonal.fit(verbose=False)
    seasonal_rmse = seasonal.evaluate(in_sample=True).loc["x", "rmse"]

    assert seasonal_rmse < baseline_rmse


def test_decomposition_period_infers_from_seasonality():
    assert decomposition_period("weekly", timedelta(days=1)) == 7
    assert decomposition_period("monthly", timedelta(days=1)) == 30
    assert decomposition_period("yearly", timedelta(weeks=1)) == 52


def test_cyclical_features_added():
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"timestamp": dates, "x": range(10)})
    df, cols = add_cyclical_time_features(df, "timestamp")
    assert "month_sin" in cols
    assert len(df) == 10
