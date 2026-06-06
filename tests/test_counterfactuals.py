import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_forecast import CausalForecaster


def _make_forecaster():
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": np.random.normal(25, 3, len(dates)),
            "crop_yield": np.random.normal(100, 5, len(dates)),
        }
    )
    graph = nx.DiGraph()
    graph.add_edge("temperature", "crop_yield")

    forecaster = CausalForecaster(
        data,
        graph,
        "crop_yield",
        "timestamp",
        forecast_horizon=5,
        lookback_periods=3,
    )
    forecaster.fit(verbose=False)
    return forecaster


def test_scalar_counterfactual_applies_to_all_steps():
    forecaster = _make_forecaster()
    steps = 5
    result = forecaster.run_counterfactual({"temperature": 40.0}, steps=steps)

    assert len(result) == steps
    assert (result["temperature"] == 40.0).all()


def test_list_counterfactual_per_horizon():
    forecaster = _make_forecaster()
    steps = 5
    values = [30.0, 31.0, 32.0, 33.0, 34.0]
    result = forecaster.run_counterfactual({"temperature": values}, steps=steps)

    assert list(result["temperature"]) == values


def test_list_counterfactual_wrong_length_raises():
    forecaster = _make_forecaster()

    with pytest.raises(ValueError, match="expected 5"):
        forecaster.run_counterfactual({"temperature": [30.0, 31.0]}, steps=5)
