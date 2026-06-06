import networkx as nx
import numpy as np
import pandas as pd

from causal_forecast import CausalForecaster


def _make_forecaster(n=80):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    temperature = np.random.normal(25, 3, n)
    rain = (temperature < 22).astype(int)
    crop_yield = temperature + rain * 5 + np.random.normal(0, 1, n)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": temperature,
            "rain": rain,
            "crop_yield": crop_yield,
        }
    )

    graph = nx.DiGraph()
    graph.add_edges_from([("temperature", "rain"), ("rain", "crop_yield")])

    return CausalForecaster(
        data,
        graph,
        "crop_yield",
        "timestamp",
        forecast_horizon=5,
        lookback_periods=3,
    )


def test_backtest_no_future_data_in_training():
    forecaster = _make_forecaster()
    forecaster.fit(verbose=False)

    horizon = 5
    min_train = 30
    step_size = 10

    backtest_df = forecaster.backtest(
        horizon=horizon,
        min_train_size=min_train,
        step_size=step_size,
    )

    assert not backtest_df.empty
    assert set(backtest_df.columns) >= {
        "fold",
        "train_end",
        "horizon_step",
        "variable",
        "actual",
        "predicted",
    }

    for fold in backtest_df["fold"].unique():
        train_end_idx = fold - 1
        test_start_idx = fold
        test_end_idx = fold + horizon - 1

        assert train_end_idx < test_start_idx
        assert test_end_idx < len(forecaster.data)

        fold_rows = backtest_df[backtest_df["fold"] == fold]
        assert len(fold_rows) == horizon * len(forecaster.graph.nodes())


def test_backtest_does_not_mutate_fitted_models():
    forecaster = _make_forecaster()
    forecaster.fit(verbose=False)
    original_models = list(forecaster.models.keys())

    forecaster.backtest(horizon=3, min_train_size=20, step_size=15)

    assert list(forecaster.models.keys()) == original_models
