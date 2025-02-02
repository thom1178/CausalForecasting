# Causal Forecast

A Python package for causal forecasting that combines structural causal models with time series analysis. This package allows you to:
- Build and analyze causal relationships in time series data
- Make forecasts that respect causal structure
- Run counterfactual scenarios for what-if analysis
- Visualize temporal patterns and causal effects

## Installation

### From PyPI (Coming Soon)
```bash
pip install causal-forecast
```

### From Source
```bash
git clone https://github.com/thom1178/CausalForecasting.git
cd causal_forecast
pip install -e .
```

## Quick Start

```python
import networkx as nx
import pandas as pd
from causal_forecast import CausalForecaster

# Create causal graph
G = nx.DiGraph()
G.add_edges_from([
    ('temperature', 'humidity'),
    ('temperature', 'rain'),
    ('humidity', 'crop_yield'),
    ('rain', 'crop_yield')
])

# Initialize forecaster
forecaster = CausalForecaster(
    data=your_data,
    graph=G,
    target='crop_yield',
    time_column='timestamp',
    forecast_horizon=30,
    lookback_periods=7
)

# Train models
forecaster.fit()

# Make predictions
future_predictions = forecaster.predict(steps=30)

# Run counterfactual analysis
counterfactual = forecaster.run_counterfactual({
    'temperature': 35  # What if temperature is very high?
})
```

## Features

### Time Series Forecasting
- Automatic feature engineering for temporal data
- Handles multiple time-dependent variables
- Configurable forecast horizon and lookback periods

### Causal Analysis
- Supports complex causal graphs (DAGs)
- Respects causal relationships in predictions
- Enables counterfactual scenario analysis

### Visualization
```python
from causal_forecast import (
    plot_causal_graph,
    plot_time_series,
    plot_forecast_comparison,
    plot_seasonal_decomposition,
    plot_counterfactual_timeseries
)

# Plot time series data
plot_time_series(data, 'timestamp', ['temperature', 'crop_yield'])

# Compare forecasts
plot_forecast_comparison(actual_data, predictions, 'timestamp', 'crop_yield')

# Analyze seasonality
plot_seasonal_decomposition(data, 'timestamp', 'temperature')

# Visualize counterfactuals
plot_counterfactual_timeseries(predictions, counterfactual, 'timestamp')
```

## Requirements
- Python ≥ 3.7
- networkx ≥ 2.5
- pandas ≥ 1.0.0
- numpy ≥ 1.19.0
- scikit-learn ≥ 0.24.0
- matplotlib ≥ 3.3.0
- seaborn ≥ 0.11.0
- statsmodels ≥ 0.12.0

## Documentation

### CausalForecaster Class
The main class for performing causal forecasting:

```python
CausalForecaster(
    data: pd.DataFrame,          # Input data
    graph: nx.DiGraph,          # Causal graph structure
    target: str,                # Target variable
    time_column: str,           # Time column name
    forecast_horizon: int = 1,   # Steps to forecast
    lookback_periods: int = 3    # Historical periods to use
)
```

### Key Methods
- `fit()`: Train the forecasting models
- `predict(steps: int = None)`: Make future predictions
- `run_counterfactual(interventions: dict)`: Run what-if scenarios

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Citation
If you use this package in your research, please cite:
```bibtex
@software{causal_forecast2024,
  author = {Thomas, Jamel},
  title = {Causal Forecast: A Python Package for Causal Time Series Analysis},
  year = {2024},
  url = {https://github.com/yourusername/causal_forecast}
}
```
