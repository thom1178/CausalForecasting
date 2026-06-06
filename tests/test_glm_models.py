import networkx as nx
import numpy as np
import pandas as pd

from causal_forecast import CausalForecaster


def _make_graph_and_data(n=200):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    temperature = 20 + 0.01 * t + np.random.normal(0, 1, n)
    rain = (temperature < 22).astype(int)
    region = np.random.choice(["North", "South", "East"], n)
    crop_yield = temperature + rain * 5 + np.random.normal(80, 3, n)

    data = pd.DataFrame(
        {
            "timestamp": dates,
            "temperature": temperature,
            "rain": rain,
            "region": region,
            "crop_yield": crop_yield,
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
    return graph, data


def test_glm_default_random_forest_unchanged():
    graph, data = _make_graph_and_data()
    forecaster = CausalForecaster(data, graph, "crop_yield", "timestamp", lookback_periods=3)
    forecaster.fit(verbose=False)
    assert forecaster.models["temperature"].model_backend == "random_forest"
    pred = forecaster.predict(steps=5)
    assert len(pred) == 5


def test_glm_continuous_binary_multiclass_fit_and_predict():
    graph, data = _make_graph_and_data()
    forecaster = CausalForecaster(
        data,
        graph,
        "crop_yield",
        "timestamp",
        lookback_periods=3,
        model_type="glm",
    )
    forecaster.fit(verbose=False)

    for node in graph.nodes():
        assert forecaster.models[node].model_backend == "glm"
        assert forecaster.models[node].feature_names is not None

    predictions = forecaster.predict(steps=7)
    assert len(predictions) == 7
    assert set(predictions["region"].unique()).issubset({"North", "South", "East"})
    assert set(predictions["rain"].unique()).issubset({0, 1})


def test_glm_in_sample_and_evaluate():
    graph, data = _make_graph_and_data()
    forecaster = CausalForecaster(
        data,
        graph,
        "crop_yield",
        "timestamp",
        lookback_periods=3,
        model_type="glm",
    )
    forecaster.fit(verbose=False)

    in_sample = forecaster.predict_in_sample()
    assert len(in_sample) > 0

    metrics = forecaster.evaluate(holdout_steps=14)
    assert "rmse" in metrics.columns or "accuracy" in metrics.columns


def test_glm_on_logistic_trend_produces_finite_metrics():
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    t = np.arange(n)
    logistic = 20 / (1 + np.exp(-0.02 * (t - n / 2)))
    y = logistic + np.random.normal(0, 0.5, n)

    data = pd.DataFrame({"timestamp": dates, "x": y})
    graph = nx.DiGraph()
    graph.add_node("x")

    glm = CausalForecaster(data, graph, "x", "timestamp", lookback_periods=3, model_type="glm")
    glm.fit(verbose=False)
    glm_rmse = glm.evaluate(in_sample=True).loc["x", "rmse"]

    assert np.isfinite(glm_rmse)
    assert glm.models["x"].model_backend == "glm"
