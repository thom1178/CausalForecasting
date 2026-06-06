import pandas as pd

from causal_forecast.metrics import summarize_fit_quality


def test_summarize_fit_quality_handles_missing_metric_for_mixed_types():
    metrics_df = pd.DataFrame(
        {
            "variable_type": ["continuous", "binary", "multiclass"],
            "rmse": [1.0, float("nan"), float("nan")],
            "f1": [float("nan"), 0.8, float("nan")],
            "macro_f1": [float("nan"), float("nan"), 0.5],
        },
        index=["temperature", "rain", "region"],
    )

    summary = summarize_fit_quality(metrics_df, metric="rmse")

    assert summary.loc["temperature", "fit_quality"] in ("good", "poor")
    assert summary.loc["rain", "fit_quality"] == "n/a"
    assert pd.isna(summary.loc["rain", "rank"])


def test_summarize_fit_quality_type_aware():
    metrics_df = pd.DataFrame(
        {
            "variable_type": ["continuous", "binary"],
            "rmse": [2.0, float("nan")],
            "f1": [float("nan"), 0.9],
        },
        index=["crop_yield", "rain"],
    )

    summary = summarize_fit_quality(metrics_df, use_type_aware_metric=True)

    assert len(summary) == 2
    assert "metric_used" in summary.columns
