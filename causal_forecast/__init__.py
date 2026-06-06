from .core import CausalForecaster
from .typing import VariableType, detect_variable_types
from .metrics import (
    mae,
    mse,
    rmse,
    mape,
    evaluate_forecast,
    evaluate_forecast_typed,
    evaluate_by_horizon,
    evaluate_variable,
    summarize_fit_quality,
    summarize_backtest,
    primary_metric_for_type,
)
from .viz import (
    plot_time_series,
    plot_forecast_comparison,
    plot_seasonal_decomposition,
    plot_counterfactual_timeseries,
    plot_residuals,
    plot_metrics_summary,
)

__version__ = "0.2.0"
