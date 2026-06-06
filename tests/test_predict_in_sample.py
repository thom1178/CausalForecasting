import networkx as nx
import numpy as np
import pandas as pd

from causal_forecast import CausalForecaster


def test_predict_in_sample_aligns_mixed_type_nodes():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": np.random.normal(25, 3, len(dates)),
            "rain": np.random.binomial(1, 0.3, len(dates)),
            "region": np.random.choice(["North", "South", "East"], len(dates)),
            "crop_yield": np.random.normal(100, 5, len(dates)),
        }
    )

    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("temperature", "rain"),
            ("region", "crop_yield"),
            ("rain", "crop_yield"),
        ]
    )

    forecaster = CausalForecaster(
        data,
        graph,
        "crop_yield",
        "timestamp",
        lookback_periods=5,
        use_one_hot_parents=True,
    )
    forecaster.fit(verbose=False)

    in_sample = forecaster.predict_in_sample()
    metrics = forecaster.evaluate(in_sample=True)

    assert len(in_sample) == len(data) - forecaster.lookback_periods
    assert not metrics.empty
