# Causal Forecast

A Python package for causal forecasting that combines structural causal models with time series analysis. This package allows you to:
- Build and analyze causal relationships in time series data
- Automatically detect continuous, binary, multiclass, and ordinal variables
- Make forecasts that respect causal structure with nonlinear RandomForest models
- Run counterfactual scenarios (scalar or per-horizon interventions)
- Backtest with walk-forward expanding windows (no data leakage)
- Evaluate fit quality with type-aware metrics

## Installation

### From Source
```bash
git clone https://github.com/thom1178/CausalForecasting.git
cd CausalForecasting
pip install -e causal_forecast
```

## Quick Start

```python
import networkx as nx
import pandas as pd
import numpy as np
from causal_forecast import CausalForecaster, detect_variable_types

dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.random.normal(25, 5, len(dates)),
    'humidity': np.random.normal(60, 3, len(dates)),
    'rain': np.random.binomial(1, 0.3, len(dates)),
    'crop_yield': np.random.normal(90, 10, len(dates)),
})

G = nx.DiGraph()
G.add_edges_from([
    ('temperature', 'humidity'),
    ('temperature', 'rain'),
    ('humidity', 'crop_yield'),
    ('rain', 'crop_yield'),
])

# Auto-detect variable types
print(detect_variable_types(data, G, 'timestamp'))

forecaster = CausalForecaster(
    data=data,
    graph=G,
    target='crop_yield',
    time_column='timestamp',
    forecast_horizon=30,
    lookback_periods=7,
)

forecaster.fit()
future_predictions = forecaster.predict(steps=30)

# Per-horizon counterfactual (list length must equal horizon)
counterfactual = forecaster.run_counterfactual({
    'temperature': list(np.linspace(30, 40, 30))
})

# Type-aware evaluation
metrics = forecaster.evaluate(holdout_steps=30)
print(metrics)

# Walk-forward backtest (no leakage)
backtest_results = forecaster.backtest(horizon=10, min_train_size=60, step_size=10)
summary = forecaster.summarize_backtest(backtest_results)
print(summary)
```

## Features

### Automatic Variable Type Detection
- **continuous**: numeric variables with many unique values
- **binary**: two-class flags (0/1, yes/no)
- **multiclass**: categorical strings or low-cardinality integers
- **ordinal**: ordered pandas `CategoricalDtype`

Override detection with `type_overrides={'rain': 'binary'}`.

### Type-Aware Models
| Type | Model | Metrics |
|------|-------|---------|
| continuous | RandomForestRegressor | MAE, RMSE, MAPE |
| binary | RandomForestClassifier | accuracy, F1, Brier score |
| multiclass | RandomForestClassifier | accuracy, macro-F1, log loss |
| ordinal | RandomForestClassifier | accuracy, ordinal MAE |

### Walk-Forward Backtest
`backtest()` uses an expanding training window. Each fold retrains only on data strictly before the test window. Use `n_jobs > 1` for parallel folds.

### Counterfactuals
- Scalar: `{'temperature': 35}` applies to all steps
- Per-horizon: `{'temperature': [35, 36, 37, ...]}` list length must equal `steps`

### Visualization
```python
from causal_forecast import (
    plot_time_series,
    plot_forecast_comparison,
    plot_seasonal_decomposition,
    plot_counterfactual_timeseries,
    plot_residuals,
    plot_metrics_summary,
)

plot_time_series(data, 'timestamp', ['temperature', 'crop_yield'])
plot_forecast_comparison(actual_data, predictions, 'timestamp', 'crop_yield')
plot_metrics_summary(metrics, metric='rmse')
plot_residuals(holdout_data, predictions, 'timestamp', 'crop_yield')
```

## API Reference

### CausalForecaster
```python
CausalForecaster(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    target: str,
    time_column: str,
    forecast_horizon: int = 1,
    lookback_periods: int = 3,
    type_overrides: dict = None,
    use_one_hot_parents: bool = True,
)
```

### Key Methods
- `fit(verbose=True)`: Train type-aware models per node
- `predict(steps, counterfactuals)`: Multi-step forecast
- `run_counterfactual(interventions, steps)`: What-if scenarios
- `evaluate(holdout_steps, in_sample=False)`: Type-aware metrics
- `backtest(horizon, min_train_size, step_size, n_jobs=1)`: Walk-forward backtest
- `summarize_backtest(backtest_df)`: Aggregate backtest results
- `detect_variable_types(data, graph, time_column)`: Standalone type detection

## Requirements
- Python ≥ 3.7
- networkx ≥ 2.5
- pandas ≥ 1.0.0
- numpy ≥ 1.19.0
- scikit-learn ≥ 0.24.0
- matplotlib ≥ 3.3.0
- seaborn ≥ 0.11.0
- statsmodels ≥ 0.12.0
- joblib ≥ 1.0.0

## License
MIT License
