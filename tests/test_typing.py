import networkx as nx
import numpy as np
import pandas as pd
import pytest

from causal_forecast import detect_variable_types, CausalForecaster


def _make_data():
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": np.random.normal(25, 3, len(dates)),
            "rain": np.random.binomial(1, 0.3, len(dates)),
            "region": np.random.choice(["North", "South", "East"], len(dates)),
            "risk": pd.Categorical(
                np.random.choice(["low", "medium", "high"], len(dates)),
                categories=["low", "medium", "high"],
                ordered=True,
            ),
            "crop_yield": np.random.normal(100, 5, len(dates)),
        }
    )


def _make_graph():
    g = nx.DiGraph()
    g.add_edges_from(
        [
            ("temperature", "rain"),
            ("region", "crop_yield"),
            ("rain", "crop_yield"),
        ]
    )
    return g


def test_detect_continuous_binary_multiclass_ordinal():
    data = _make_data()
    graph = _make_graph()
    graph.add_node("risk")

    types = detect_variable_types(data, graph, "timestamp")

    assert types["temperature"] == "continuous"
    assert types["rain"] == "binary"
    assert types["region"] == "multiclass"
    assert types["risk"] == "ordinal"


def test_type_overrides():
    data = _make_data()
    graph = _make_graph()

    types = detect_variable_types(
        data,
        graph,
        "timestamp",
        overrides={"rain": "continuous"},
    )
    assert types["rain"] == "continuous"


def test_forecaster_stores_detected_types():
    data = _make_data()
    graph = _make_graph()

    forecaster = CausalForecaster(data, graph, "crop_yield", "timestamp", lookback_periods=3)
    assert forecaster.variable_types["rain"] == "binary"
    assert forecaster.variable_types["region"] == "multiclass"
