# Causal Forecast

A Python package for **causal time-series forecasting**. Given a directed acyclic graph (DAG) of variable relationships and historical data, Causal Forecast trains one model per node, predicts in topological order, and propagates effects through the graph — supporting continuous, binary, multiclass, and ordinal variables.

**Version:** 0.2.0

## What it does

- Accepts a user-defined DAG and automatically detects variable types
- Trains per-node models with **Random Forest** (default) or **GLM** backends
- Forecasts multiple steps ahead using lag features and calendar time features
- Runs counterfactual what-if scenarios (fixed or per-horizon interventions)
- Evaluates fit quality with type-aware metrics
- Backtests with walk-forward expanding windows (no data leakage)
- Infers time frequency from data (daily, weekly, monthly, etc.)
- Optional weekly, monthly, and yearly seasonality (cyclical features + seasonal lags)

## Installation

```bash
git clone https://github.com/thom1178/CausalForecasting.git
cd CausalForecasting
pip install -e causal_forecast
```

### Requirements

- Python ≥ 3.7
- networkx ≥ 2.5
- pandas ≥ 1.0.0
- numpy ≥ 1.19.0
- scikit-learn ≥ 0.24.0
- matplotlib ≥ 3.3.0
- seaborn ≥ 0.11.0
- statsmodels ≥ 0.12.0
- joblib ≥ 1.0.0

## Quick start

```python
import networkx as nx
import numpy as np
import pandas as pd
from causal_forecast import CausalForecaster, detect_variable_types

# Mixed-type time series: continuous, binary, multiclass
dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
data = pd.DataFrame({
    "timestamp": dates,
    "temperature": np.random.normal(25, 5, len(dates)),
    "humidity": np.random.normal(60, 3, len(dates)),
    "rain": np.random.binomial(1, 0.3, len(dates)),
    "region": np.random.choice(["North", "South", "East"], len(dates)),
    "crop_yield": np.random.normal(90, 10, len(dates)),
})

G = nx.DiGraph()
G.add_edges_from([
    ("temperature", "humidity"),
    ("temperature", "rain"),
    ("region", "crop_yield"),
    ("humidity", "crop_yield"),
    ("rain", "crop_yield"),
])

print(detect_variable_types(data, G, "timestamp"))
# {'temperature': 'continuous', 'humidity': 'continuous', 'rain': 'binary',
#  'region': 'multiclass', 'crop_yield': 'continuous'}

forecaster = CausalForecaster(
    data=data,
    graph=G,
    target="crop_yield",
    time_column="timestamp",
    forecast_horizon=30,
    lookback_periods=7,
)

forecaster.fit()
predictions = forecaster.predict(steps=30)
```

See [`analytics/example.ipynb`](analytics/example.ipynb) for a full walkthrough with evaluation plots.

## How it works

1. **Type detection** — Each graph node is classified as `continuous`, `binary`, `multiclass`, or `ordinal`.
2. **Feature engineering** — Calendar features (`year`, `month`, `day`, `dayofweek`) plus short-term lags. Optionally adds cyclical sin/cos encodings and seasonal lag features (`weekly`, `monthly`, `yearly`).
3. **Per-node training** — Nodes are fit in topological order using `model_type='random_forest'` (default) or `model_type='glm'`.
4. **Multi-step prediction** — Each future step predicts all nodes in DAG order. Lag features use historical data for early steps, then prior predictions.
5. **Counterfactuals** — Intervened nodes are overridden; downstream nodes still propagate through the graph.

## Variable type detection

| Type | Detection rule | Example |
|------|----------------|---------|
| `continuous` | Numeric with many unique values | temperature, revenue |
| `binary` | Exactly two values (0/1, yes/no) | rain, churn |
| `multiclass` | Object/category dtype or low-cardinality integer | region, SKU |
| `ordinal` | Ordered pandas `CategoricalDtype` | low / medium / high |

Override auto-detection:

```python
forecaster = CausalForecaster(
    data, G, "crop_yield", "timestamp",
    type_overrides={"rain": "binary"},
)
```

Or supply types explicitly with `variable_types={...}` to skip detection entirely.

## Model backends

Choose the per-node model with `model_type`:

```python
# Default: nonlinear Random Forest
forecaster = CausalForecaster(data, G, "crop_yield", "timestamp", model_type="random_forest")

# GLM: type-appropriate generalized linear models via statsmodels
forecaster = CausalForecaster(data, G, "crop_yield", "timestamp", model_type="glm")
```

| Variable type | Random Forest | GLM |
|---------------|---------------|-----|
| continuous | `RandomForestRegressor` | Gaussian GLM |
| binary | `RandomForestClassifier` | Binomial GLM (logit) |
| multiclass | `RandomForestClassifier` | Multinomial logit |
| ordinal | `RandomForestClassifier` | Ordered logit (falls back to multinomial) |

GLM is useful when relationships are approximately linear on the link scale (e.g. logistic trends). Random Forest remains the default for complex nonlinear patterns.

## Seasonality (opt-in)

Seasonality is **disabled by default** for backward compatibility. Enable explicitly:

```python
forecaster = CausalForecaster(
    data, G, "crop_yield", "timestamp",
    lookback_periods=7,
    seasonality=["weekly", "monthly", "yearly"],
)
```

When enabled, the model adds:
- **Cyclical features** — `dayofweek_sin/cos`, `month_sin/cos`, `dayofyear_sin/cos`
- **Seasonal lag features** — `{var}_s_lag_weekly`, `{var}_s_lag_monthly`, `{var}_s_lag_yearly`

Seasonal lag periods depend on inferred data frequency:

| Data frequency | weekly | monthly | yearly |
|----------------|--------|---------|--------|
| daily | 7 steps | 30 steps | 365 steps |
| weekly | — | 4 steps | 52 steps |
| monthly | — | — | 12 steps |

Minimum data length: `period + lookback_periods + 1` rows (e.g. 373 rows for yearly seasonality with `lookback_periods=7` on daily data).

Decomposition plots accept a seasonality type:

```python
plot_seasonal_decomposition(data, "timestamp", "temperature", seasonality="weekly")
plot_seasonal_decomposition(data, "timestamp", "temperature", seasonality="monthly")
plot_seasonal_decomposition(data, "timestamp", "temperature", seasonality="yearly")
```

## Forecasting and counterfactuals

```python
# Baseline forecast
predictions = forecaster.predict(steps=30)

# Scalar intervention — same value every step
cf_fixed = forecaster.run_counterfactual({"temperature": 35}, steps=30)

# Per-horizon intervention — list length must equal steps
cf_ramp = forecaster.run_counterfactual(
    {"temperature": list(np.linspace(30, 40, 30))},
    steps=30,
)
```

## Evaluation

### Holdout and in-sample

```python
from causal_forecast import summarize_fit_quality

# Out-of-sample: refit on train, predict holdout (no leakage)
metrics, preds, actual = forecaster.evaluate(
    holdout_steps=30,
    return_predictions=True,
)

# In-sample: per-node model fit on training rows
in_sample = forecaster.evaluate(in_sample=True)

# Per-horizon error for one variable
horizon_errors = forecaster.evaluate_horizon(variable="crop_yield", holdout_steps=30)
```

### Type-aware metrics

| Variable type | Metrics |
|---------------|---------|
| continuous | MAE, MSE, RMSE, MAPE |
| binary | accuracy, F1, Brier score |
| multiclass | accuracy, macro-F1, log loss |
| ordinal | accuracy, ordinal MAE |

Summarize fit quality across mixed-type graphs:

```python
# Rank each variable using its type-appropriate metric
summarize_fit_quality(metrics, use_type_aware_metric=True)

# Or rank continuous variables only by RMSE
summarize_fit_quality(metrics, metric="rmse")
```

### Walk-forward backtest

```python
backtest_results = forecaster.backtest(
    horizon=10,
    min_train_size=60,
    step_size=15,
    n_jobs=1,  # set > 1 for parallel folds
)

summary = forecaster.summarize_backtest(backtest_results)
```

Each fold:
- Trains only on `data[:cutoff]` (strict past)
- Predicts the next `horizon` steps
- Does not modify the caller's fitted models

## Visualization

```python
from causal_forecast import (
    plot_time_series,
    plot_forecast_comparison,
    plot_seasonal_decomposition,
    plot_counterfactual_timeseries,
    plot_residuals,
    plot_metrics_summary,
)

plot_time_series(data, "timestamp", ["temperature", "crop_yield"])
plot_forecast_comparison(actual, predictions, "timestamp", "crop_yield")
plot_counterfactual_timeseries(predictions, cf_ramp, "timestamp")
plot_metrics_summary(metrics, metric="rmse")
plot_residuals(actual, predictions, "timestamp", "crop_yield")
```

## API reference

### `CausalForecaster`

```python
CausalForecaster(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    target: str,
    time_column: str,
    forecast_horizon: int = 1,
    lookback_periods: int = 3,
    variable_types: dict = None,      # skip auto-detection
    type_overrides: dict = None,      # override specific nodes
    use_one_hot_parents: bool = True, # one-hot encode multiclass/ordinal parent lags
    seasonality: list = None,        # opt-in: ['weekly', 'monthly', 'yearly']
    model_type: str = "random_forest",  # or "glm"
)
```

| Method | Description |
|--------|-------------|
| `fit(verbose=True)` | Train type-aware models for each node |
| `predict(steps, counterfactuals, return_proba=False)` | Multi-step forecast |
| `predict_in_sample()` | One-step-ahead predictions on training data |
| `run_counterfactual(interventions, steps)` | What-if scenario forecasting |
| `evaluate(holdout_steps, in_sample, return_predictions)` | Type-aware accuracy metrics |
| `evaluate_horizon(variable, holdout_steps)` | Per-step error for one variable |
| `backtest(horizon, min_train_size, step_size, n_jobs)` | Walk-forward backtest |
| `summarize_backtest(backtest_df, metric)` | Aggregate backtest folds |
| `summarize_models(detailed=False)` | Overview of each fitted node model |
| `summarize_model(node)` | GLM coefficients or RF feature importances for one node |

### Standalone utilities

```python
from causal_forecast import (
    detect_variable_types,
    infer_data_frequency,
    seasonal_periods,
    decomposition_period,
    evaluate_forecast,
    evaluate_forecast_typed,
    evaluate_variable,
    summarize_fit_quality,
    summarize_backtest,
    primary_metric_for_type,
    mae, mse, rmse, mape,
)
```

| Function | Description |
|----------|-------------|
| `detect_variable_types(data, graph, time_column, overrides)` | Auto-detect node types |
| `infer_data_frequency(time_delta)` | Map timestep to daily/weekly/monthly/yearly |
| `seasonal_periods(frequency, names)` | Lag step counts per seasonality type |
| `decomposition_period(seasonality, time_delta)` | Period for statsmodels decomposition |
| `evaluate_forecast_typed(actual, predicted, time_column, variables, variable_types)` | Metrics per variable type |
| `summarize_fit_quality(metrics_df, metric, use_type_aware_metric)` | Rank good vs poor fits |
| `summarize_backtest(backtest_df, metric, variable_types)` | Aggregate fold results |
| `forecaster.summarize_models(detailed=False)` | Overview of each fitted node model |
| `forecaster.summarize_model(node)` | Coefficients (GLM) or importances (RF) for one node |
| `primary_metric_for_type(variable_type)` | Default metric name per type |

## Project structure

```
CausalForecasting/
├── causal_forecast/
│   ├── core.py        # CausalForecaster
│   ├── typing.py      # Variable type detection
│   ├── utils.py       # Model training and encoding
│   ├── metrics.py     # Evaluation and backtest summaries
│   ├── seasonality.py # Seasonal features and period mapping
│   ├── glm_models.py  # GLM training and prediction
│   └── viz.py         # Plotting helpers
├── analytics/
│   └── example.ipynb  # End-to-end demo
├── tests/
└── README.md
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Contributing

Contributions are welcome. Please open an issue or pull request on [GitHub](https://github.com/thom1178/CausalForecasting).

## License

MIT License
