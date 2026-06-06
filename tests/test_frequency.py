import networkx as nx
import numpy as np
import pandas as pd

from causal_forecast import CausalForecaster
from causal_forecast.utils import infer_time_delta


def test_infer_weekly_frequency():
    dates = pd.date_range("2023-01-01", periods=10, freq="W")
    delta = infer_time_delta(dates)
    assert delta.days == 7


def test_predict_uses_inferred_frequency():
    dates = pd.date_range("2023-01-01", periods=30, freq="W")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "x": np.random.normal(0, 1, len(dates)),
            "y": np.random.normal(0, 1, len(dates)),
        }
    )
    graph = nx.DiGraph()
    graph.add_edge("x", "y")

    forecaster = CausalForecaster(data, graph, "y", "timestamp", lookback_periods=2)
    forecaster.fit(verbose=False)
    pred = forecaster.predict(steps=2)

    expected_dates = [
        dates[-1] + pd.Timedelta(weeks=1),
        dates[-1] + pd.Timedelta(weeks=2),
    ]
    assert pred["timestamp"].tolist() == expected_dates
