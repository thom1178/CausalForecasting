import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime

from statsmodels.tsa.seasonal import seasonal_decompose

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
    figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Plot seasonal decomposition of time series.
    
    Args:
        data: DataFrame containing time series data
        time_column: Name of the time column
        variable: Variable to decompose
        figsize: Figure size
    """
    
    
    # Convert to datetime index
    ts_data = data.set_index(time_column)[variable]
    
    # Perform decomposition
    decomposition = seasonal_decompose(ts_data, period=30)
    
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=figsize)
    
    # Plot original
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Original Time Series')
    
    # Plot trend
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    
    # Plot seasonal
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    
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

