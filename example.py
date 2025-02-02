import networkx as nx
import pandas as pd
import numpy as np
from causal_forecast import CausalForecaster

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)


# Create causal graph
G = nx.DiGraph()
G.add_edges_from([
    ('temperature', 'humidity'),
    ('temperature', 'rain'),
    ('humidity', 'crop_yield'),
    ('rain', 'crop_yield')
])


# Create sample time series data
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
data = pd.DataFrame({
    'timestamp': dates,
    'temperature': np.random.normal(25, 5, len(dates)),
    'humidity': np.random.normal(60, 10, len(dates)),
    'rain': np.random.binomial(1, 0.3, len(dates)),
    'crop_yield': np.random.normal(90, 10, len(dates))
})

# Initialize forecaster with time component
forecaster = CausalForecaster(
    data=data,
    graph=G,
    target='crop_yield',
    time_column='timestamp',
    forecast_horizon=30,  # Forecast 30 days ahead
    lookback_periods=7    # Use 7 days of history
)

# Train the models
forecaster.fit()

# Make time series predictions
future_predictions = forecaster.predict(steps=30)
print("Future predictions:\n", future_predictions.head())

# Run counterfactual scenario
counterfactual_predictions = forecaster.run_counterfactual(
    interventions={'temperature': 35},
    steps=30
)
print("\nCounterfactual predictions:\n", counterfactual_predictions.head())