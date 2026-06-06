import numpy as np
import pandas as pd
from typing import List, Tuple, Union


def mae(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mse(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Mean Squared Error."""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def rmse(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: Union[pd.Series, np.ndarray], y_pred: Union[pd.Series, np.ndarray]) -> float:
    """Mean Absolute Percentage Error (returns NaN if any true value is zero)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.any(y_true == 0):
        return float("nan")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def _align_actual_and_predicted(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    variables: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge actual and predicted DataFrames on the time column."""
    actual_time = time_column if time_column in actual.columns else time_column
    pred_time = time_column if time_column in predicted.columns else "timestamp"

    merged = actual[[actual_time] + variables].merge(
        predicted[[pred_time] + variables],
        left_on=actual_time,
        right_on=pred_time,
        suffixes=("_actual", "_pred"),
    )
    if pred_time != actual_time and pred_time in merged.columns:
        merged = merged.drop(columns=[pred_time])

    actual_aligned = merged[[f"{v}_actual" for v in variables]].rename(
        columns={f"{v}_actual": v for v in variables}
    )
    predicted_aligned = merged[[f"{v}_pred" for v in variables]].rename(
        columns={f"{v}_pred": v for v in variables}
    )
    return actual_aligned, predicted_aligned


def evaluate_forecast(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    variables: List[str],
) -> pd.DataFrame:
    """
    Compute error metrics for each variable.

    Returns a DataFrame indexed by variable with columns: mae, mse, rmse, mape.
    """
    actual_aligned, predicted_aligned = _align_actual_and_predicted(
        actual, predicted, time_column, variables
    )

    results = []
    for var in variables:
        y_true = actual_aligned[var]
        y_pred = predicted_aligned[var]
        results.append(
            {
                "variable": var,
                "mae": mae(y_true, y_pred),
                "mse": mse(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "mape": mape(y_true, y_pred),
            }
        )

    return pd.DataFrame(results).set_index("variable")


def evaluate_by_horizon(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    variable: str,
) -> pd.DataFrame:
    """
    Compute metrics at each forecast step (horizon).

    Useful for seeing how error compounds over multi-step forecasts.
    """
    actual_time = time_column if time_column in actual.columns else time_column
    pred_time = time_column if time_column in predicted.columns else "timestamp"

    merged = actual[[actual_time, variable]].merge(
        predicted[[pred_time, variable]],
        left_on=actual_time,
        right_on=pred_time,
        suffixes=("_actual", "_pred"),
    )
    if pred_time != actual_time and pred_time in merged.columns:
        merged = merged.drop(columns=[pred_time])

    merged = merged.reset_index(drop=True)
    merged["horizon"] = range(1, len(merged) + 1)
    merged["error"] = merged[f"{variable}_pred"] - merged[f"{variable}_actual"]
    merged["abs_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2

    return merged[
        [
            "horizon",
            actual_time,
            f"{variable}_actual",
            f"{variable}_pred",
            "error",
            "abs_error",
            "squared_error",
        ]
    ]


def summarize_fit_quality(metrics_df: pd.DataFrame, metric: str = "rmse") -> pd.DataFrame:
    """
    Rank variables by fit quality and label good vs poor fits.

    Variables below the median metric are labeled 'good', above are 'poor'.
    """
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(metrics_df.columns)}")

    summary = metrics_df[[metric]].copy()
    median_value = summary[metric].median()
    summary["fit_quality"] = np.where(
        summary[metric] <= median_value, "good", "poor"
    )
    summary["rank"] = summary[metric].rank().astype(int)
    return summary.sort_values(metric)
