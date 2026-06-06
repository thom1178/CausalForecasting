import networkx as nx
import numpy as np
import pandas as pd

from causal_forecast import CausalForecaster, summarize_models, summarize_node_model


def _fit_forecaster(model_type="random_forest"):
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    data = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": np.random.normal(25, 3, len(dates)),
            "rain": np.random.binomial(1, 0.3, len(dates)),
            "crop_yield": np.random.normal(100, 5, len(dates)),
        }
    )
    graph = nx.DiGraph()
    graph.add_edges_from([("temperature", "rain"), ("rain", "crop_yield")])

    forecaster = CausalForecaster(
        data,
        graph,
        "crop_yield",
        "timestamp",
        lookback_periods=3,
        model_type=model_type,
    )
    forecaster.fit(verbose=False)
    return forecaster


def test_summarize_models_requires_fit():
    graph = nx.DiGraph()
    graph.add_node("x")
    data = pd.DataFrame({"timestamp": pd.date_range("2023-01-01", periods=10), "x": range(10)})
    forecaster = CausalForecaster(data, graph, "x", "timestamp")
    try:
        forecaster.summarize_models()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_summarize_random_forest_models():
    forecaster = _fit_forecaster("random_forest")
    overview = forecaster.summarize_models()

    assert len(overview) == 3
    assert "top_importance" in overview.columns
    assert overview.loc["temperature", "model_backend"] == "random_forest"

    detail = forecaster.summarize_model("temperature")
    assert "importance" in detail.columns
    assert detail.iloc[0]["node"] == "temperature"


def test_summarize_glm_models():
    forecaster = _fit_forecaster("glm")
    overview, details = forecaster.summarize_models(detailed=True)

    assert len(overview) == 3
    assert overview.loc["rain", "model_backend"] == "glm"
    assert "temperature" in details
    assert "coef" in details["temperature"].columns


def test_summarize_models_standalone_function():
    forecaster = _fit_forecaster("random_forest")
    overview = summarize_models(
        forecaster.models,
        forecaster.variable_types,
        forecaster.node_order,
    )
    assert not overview.empty

    detail = summarize_node_model(forecaster.models["crop_yield"], "crop_yield")
    assert "feature" in detail.columns
