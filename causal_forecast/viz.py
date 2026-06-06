import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime

from statsmodels.tsa.seasonal import seasonal_decompose

from .seasonality import decomposition_period
from .utils import infer_time_delta

def plot_time_series(
    data: pd.DataFrame,
    time_column: str,
    variables: List[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Plot time series data for specified variables.
    
    Args:
        data: DataFrame containing time series data
        time_column: Name of the time column
        variables: List of variables to plot (defaults to all numeric columns)
        figsize: Figure size
    """
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for var in variables:
        ax.plot(data[time_column], data[var], label=var, marker='.')
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Time Series Plot')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_forecast_comparison(
    actual: pd.DataFrame,
    forecast: pd.DataFrame,
    time_column: str,
    variable: str,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Compare actual vs forecasted values.
    
    Args:
        actual: DataFrame with actual values
        forecast: DataFrame with forecasted values
        time_column: Name of the time column
        variable: Variable to plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(actual[time_column], actual[variable], 
            label='Actual', color='blue', marker='.')
    
    # Plot forecasted values
    ax.plot(forecast[time_column], forecast[variable], 
            label='Forecast', color='red', marker='.',
            linestyle='--')
    
    ax.set_xlabel('Time')
    ax.set_ylabel(variable)
    ax.set_title(f'Actual vs Forecast: {variable}')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig

def plot_seasonal_decomposition(
    data: pd.DataFrame,
    time_column: str,
    variable: str,
    figsize: tuple = (12, 10),
    period: int = None,
    seasonality: str = "weekly",
) -> plt.Figure:
    """
    Plot seasonal decomposition of time series.

    Args:
        data: DataFrame containing time series data
        time_column: Name of the time column
        variable: Variable to decompose
        figsize: Figure size
        period: Decomposition period override (in timesteps)
        seasonality: Seasonality type ('weekly', 'monthly', 'yearly') used when period is None
    """
    ts_data = data.set_index(time_column)[variable]

    if period is None:
        time_delta = infer_time_delta(data[time_column])
        period = decomposition_period(seasonality=seasonality, time_delta=time_delta)

    decomposition = seasonal_decompose(ts_data, period=period)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
    
    # Plot original
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Original Time Series')
    
    # Plot trend
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    
    # Plot seasonal
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title(f'Seasonal (period={period})')
    
    # Plot residual
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    plt.tight_layout()
    return fig

def plot_counterfactual_timeseries(
    actual: pd.DataFrame,
    counterfactual: pd.DataFrame,
    time_column: str,
    variables: List[str] = None,
    figsize: tuple = (12, 6)
) -> plt.Figure:
    """
    Compare actual vs counterfactual predictions over time.
    
    Args:
        actual: DataFrame with actual predictions
        counterfactual: DataFrame with counterfactual predictions
        time_column: Name of the time column
        variables: List of variables to plot
        figsize: Figure size
    """
    if variables is None:
        variables = actual.select_dtypes(include=[np.number]).columns.tolist()
        variables = [v for v in variables if v != time_column]
    
    n_vars = len(variables)
    fig, axes = plt.subplots(n_vars, 1, figsize=(figsize[0], figsize[1] * n_vars))
    if n_vars == 1:
        axes = [axes]
    
    for ax, var in zip(axes, variables):
        ax.plot(actual[time_column], actual[var], 
                label='Actual', color='blue', marker='.')
        ax.plot(counterfactual[time_column], counterfactual[var], 
                label='Counterfactual', color='red', marker='.', 
                linestyle='--')
        
        ax.set_title(f'{var}: Actual vs Counterfactual')
        ax.legend()
        plt.setp(ax.xaxis.get_ticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def plot_residuals(
    actual: pd.DataFrame,
    predicted: pd.DataFrame,
    time_column: str,
    variable: str,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot prediction residuals over time to highlight periods of poor fit.
    """
    pred_time = time_column if time_column in predicted.columns else "timestamp"

    merged = actual[[time_column, variable]].merge(
        predicted[[pred_time, variable]],
        left_on=time_column,
        right_on=pred_time,
        suffixes=("_actual", "_pred"),
    )
    if pred_time != time_column and pred_time in merged.columns:
        merged = merged.drop(columns=[pred_time])

    merged["residual"] = merged[f"{variable}_pred"] - merged[f"{variable}_actual"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax1.plot(merged[time_column], merged[f"{variable}_actual"], label="Actual", marker=".")
    ax1.plot(merged[time_column], merged[f"{variable}_pred"], label="Predicted", linestyle="--", marker=".")
    ax1.set_title(f"Actual vs Predicted: {variable}")
    ax1.legend()

    ax2.bar(merged[time_column], merged["residual"], width=0.8, color="steelblue", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Residuals (Predicted - Actual)")
    ax2.set_xlabel("Time")

    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_metrics_summary(
    metrics_df: pd.DataFrame,
    metric: str = "rmse",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Bar chart of per-variable metrics to compare fit quality across nodes.
    """
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(metrics_df.columns)}")

    sorted_metrics = metrics_df[metric].sort_values()
    colors = ["#2ecc71" if v <= sorted_metrics.median() else "#e74c3c" for v in sorted_metrics]

    fig, ax = plt.subplots(figsize=figsize)
    sorted_metrics.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(sorted_metrics.median(), color="gray", linestyle="--", label=f"Median {metric}")
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Per-Variable Fit Quality ({metric.upper()})")
    ax.legend()
    plt.tight_layout()
    return fig

