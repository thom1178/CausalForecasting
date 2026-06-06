import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
)

from .typing import VariableType


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
    actual_time = time_column
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


def _to_numeric_labels(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float).values
    return pd.Categorical(series).codes.astype(float)


def evaluate_variable(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    variable_type: VariableType,
) -> Dict[str, float]:
    """Compute type-appropriate metrics for a single variable."""
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    if variable_type == "continuous":
        return {
            "mae": mae(y_true, y_pred),
            "mse": mse(y_true, y_pred),
            "rmse": rmse(y_true, y_pred),
            "mape": mape(y_true, y_pred),
        }

    y_true_str = y_true.astype(str)
    y_pred_str = y_pred.astype(str)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_str, y_pred_str)),
    }

    if variable_type == "binary":
        avg = "binary"
        pos_label = y_true_str.mode().iloc[0] if len(y_true_str.mode()) else "1"
        try:
            metrics["f1"] = float(f1_score(y_true_str, y_pred_str, average=avg, pos_label=pos_label))
        except ValueError:
            metrics["f1"] = float("nan")

        try:
            y_true_bin = _to_numeric_labels(y_true)
            y_pred_bin = _to_numeric_labels(y_pred)
            if set(np.unique(y_true_bin)) <= {0.0, 1.0}:
                metrics["brier_score"] = float(brier_score_loss(y_true_bin, y_pred_bin))
        except ValueError:
            metrics["brier_score"] = float("nan")

    elif variable_type == "multiclass":
        try:
            metrics["macro_f1"] = float(
                f1_score(y_true_str, y_pred_str, average="macro", zero_division=0)
            )
        except ValueError:
            metrics["macro_f1"] = float("nan")

        labels = sorted(y_true_str.unique())
        try:
            y_true_codes = pd.Categorical(y_true, categories=labels).codes
            y_pred_codes = pd.Categorical(y_pred, categories=labels).codes
            proba = np.eye(len(labels))[y_pred_codes]
            metrics["log_loss"] = float(log_loss(y_true_codes, proba, labels=list(range(len(labels)))))
        except ValueError:
            metrics["log_loss"] = float("nan")

    elif variable_type == "ordinal":
        y_true_ord = _to_numeric_labels(y_true)
        y_pred_ord = _to_numeric_labels(y_pred)
        metrics["ordinal_mae"] = float(mean_absolute_error(y_true_ord, y_pred_ord))
        metrics["accuracy"] = float(accuracy_score(y_true_str, y_pred_str))

    return metrics


def evaluate_forecast(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    variables: List[str],
) -> pd.DataFrame:
    """
    Compute continuous-style error metrics for each variable.

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


def evaluate_forecast_typed(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    variables: List[str],
    variable_types: Dict[str, VariableType],
) -> pd.DataFrame:
    """Compute type-aware metrics for each variable."""
    actual_aligned, predicted_aligned = _align_actual_and_predicted(
        actual, predicted, time_column, variables
    )

    rows = []
    for var in variables:
        row = {"variable": var, "variable_type": variable_types[var]}
        row.update(
            evaluate_variable(
                actual_aligned[var],
                predicted_aligned[var],
                variable_types[var],
            )
        )
        rows.append(row)

    return pd.DataFrame(rows).set_index("variable")


def evaluate_by_horizon(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    variable: str,
) -> pd.DataFrame:
    """Compute metrics at each forecast step (horizon)."""
    actual_time = time_column
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


def primary_metric_for_type(variable_type: VariableType) -> str:
    """Return the default evaluation metric for a variable type."""
    return {
        "continuous": "rmse",
        "binary": "f1",
        "multiclass": "macro_f1",
        "ordinal": "ordinal_mae",
    }[variable_type]


def summarize_fit_quality(
    metrics_df: pd.DataFrame,
    metric: str = "rmse",
    use_type_aware_metric: bool = False,
) -> pd.DataFrame:
    """
    Rank variables by fit quality and label good vs poor fits.

    When metrics contain mixed variable types, some rows may not have the
    requested metric (e.g. binary nodes lack rmse). Those rows are labeled
    ``n/a`` and excluded from ranking. Set ``use_type_aware_metric=True``
    to rank each variable using its type-appropriate primary metric.
    """
    if use_type_aware_metric:
        if "variable_type" not in metrics_df.columns:
            raise ValueError("variable_type column required when use_type_aware_metric=True")

        rows = []
        for var, row in metrics_df.iterrows():
            var_metric = primary_metric_for_type(row["variable_type"])
            if var_metric not in metrics_df.columns:
                continue
            value = row[var_metric]
            if pd.isna(value):
                continue
            rows.append({"variable": var, "metric_used": var_metric, "value": value})

        if not rows:
            return pd.DataFrame(columns=["metric_used", "value", "fit_quality", "rank"])

        summary = pd.DataFrame(rows).set_index("variable")
        median_value = summary["value"].median()
        summary["fit_quality"] = np.where(summary["value"] <= median_value, "good", "poor")
        summary["rank"] = summary["value"].rank()
        return summary.sort_values("value")

    if metric not in metrics_df.columns:
        available = [c for c in metrics_df.columns if c not in ("variable_type",)]
        raise ValueError(f"Metric '{metric}' not found. Available: {available}")

    extra_cols = ["variable_type"] if "variable_type" in metrics_df.columns else []
    summary = metrics_df[[metric] + extra_cols].copy()
    valid = summary[metric].notna()

    if not valid.any():
        summary["fit_quality"] = "n/a"
        summary["rank"] = np.nan
        return summary

    median_value = summary.loc[valid, metric].median()
    summary["fit_quality"] = "n/a"
    summary.loc[valid, "fit_quality"] = np.where(
        summary.loc[valid, metric] <= median_value,
        "good",
        "poor",
    )
    summary["rank"] = np.nan
    summary.loc[valid, "rank"] = summary.loc[valid, metric].rank()
    return summary.sort_values(metric, na_position="last")


def summarize_backtest(
    backtest_df: pd.DataFrame,
    metric: str = "rmse",
    variable_types: Optional[Dict[str, VariableType]] = None,
) -> pd.DataFrame:
    """
    Aggregate walk-forward backtest results by variable and horizon step.

    Returns mean and std of per-fold errors.
    """
    if backtest_df.empty:
        return pd.DataFrame()

    variable_types = variable_types or {}
    rows = []

    for var in backtest_df["variable"].unique():
        var_df = backtest_df[backtest_df["variable"] == var]
        var_type = variable_types.get(var, var_df["variable_type"].iloc[0] if "variable_type" in var_df else "continuous")

        fold_metrics = []
        for fold in var_df["fold"].unique():
            fold_slice = var_df[var_df["fold"] == fold]
            fold_metrics.append(
                evaluate_variable(fold_slice["actual"], fold_slice["predicted"], var_type)
            )

        metric_df = pd.DataFrame(fold_metrics)
        row = {
            "variable": var,
            "variable_type": var_type,
            "n_folds": len(metric_df),
        }
        for col in metric_df.columns:
            row[f"{col}_mean"] = metric_df[col].mean()
            row[f"{col}_std"] = metric_df[col].std()

        rows.append(row)

    summary = pd.DataFrame(rows).set_index("variable")

    if metric in summary.columns or f"{metric}_mean" in summary.columns:
        sort_col = metric if metric in summary.columns else f"{metric}_mean"
        summary = summary.sort_values(sort_col)

    return summary
