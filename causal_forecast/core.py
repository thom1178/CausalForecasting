import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta
from .utils import validate_graph_data, train_node_model

class CausalForecaster:
    """
    A causal forecasting class that combines structural causal models with time series forecasting.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        graph: nx.DiGraph,
        target: str,
        time_column: str,
        forecast_horizon: int = 1,
        lookback_periods: int = 3
    ):
        """
        Initialize the CausalForecaster.
        
        Args:
            data: Input DataFrame containing all variables
            graph: NetworkX DiGraph representing causal relationships
            target: Name of the target variable to forecast
            time_column: Name of the time column
            forecast_horizon: Number of periods to forecast ahead
            lookback_periods: Number of historical periods to use for prediction
        """
        self.data = data.copy()
        self.graph = graph.copy()
        self.target = target
        self.time_column = time_column
        self.forecast_horizon = forecast_horizon
        self.lookback_periods = lookback_periods
        
        # Validate inputs
        validate_graph_data(self.data, self.graph, self.target)
        
        # Sort data by time
        self.data = self.data.sort_values(time_column)
        
        # Initialize models dictionary
        self.models: Dict[str, RandomForestRegressor] = {}
        
        # Get topological order of nodes
        self.node_order = list(nx.topological_sort(self.graph))
        
    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from the time column."""
        df = df.copy()
        df['year'] = pd.to_datetime(df[self.time_column]).dt.year
        df['month'] = pd.to_datetime(df[self.time_column]).dt.month
        df['day'] = pd.to_datetime(df[self.time_column]).dt.day
        df['dayofweek'] = pd.to_datetime(df[self.time_column]).dt.dayofweek
        return df
    
    def _prepare_time_series_data(self, node: str) -> tuple:
        """Prepare time series data with lagged features."""
        df = self._create_time_features(self.data)
        
        # Create lagged features for each parent node
        parents = list(self.graph.predecessors(node))
        features = []
        
        # Add time features
        time_features = ['year', 'month', 'day', 'dayofweek']
        features.extend(time_features)
        
        # Add lagged features for parents and the node itself
        all_vars = parents + ([node] if node not in parents else [])
        for var in all_vars:
            for lag in range(1, self.lookback_periods + 1):
                df[f'{var}_lag_{lag}'] = df[var].shift(lag)
                features.append(f'{var}_lag_{lag}')
        
        # Drop rows with NaN values from lagging
        df = df.dropna()
        
        X = df[features]
        y = df[node]
        
        return X, y
    
    def fit(self):
        """Train models for each node in the causal graph."""
        for node in self.node_order:
            print(f"Training model for node: {node}")

            X, y = self._prepare_time_series_data(node)
            self.models[node] = train_node_model(X, y)
    
    def predict(self, steps: int = None, counterfactuals: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Make time series predictions using the causal model.
        """
        steps = steps or self.forecast_horizon
        predictions = []
        last_date = pd.to_datetime(self.data[self.time_column].max())
        
        # Create future dates
        future_dates = [last_date + timedelta(days=i+1) for i in range(steps)]
        
        # Get the last lookback_periods of actual data
        historical_data = self.data.copy().tail(self.lookback_periods)
        
        for future_date in future_dates:
            pred_row = {'timestamp': future_date}
            
            # Create time features for prediction
            time_features = {
                'year': future_date.year,
                'month': future_date.month,
                'day': future_date.day,
                'dayofweek': future_date.dayofweek
            }
            
            # Predict each node in topological order
            for node in self.node_order:
                if counterfactuals and node in counterfactuals:
                    pred_row[node] = counterfactuals[node]
                else:
                    # Get parent nodes
                    parents = list(self.graph.predecessors(node))
                    features = {}
                    
                    # Add time features
                    features.update(time_features)
                    
                    # Add lagged features for parents and the node itself
                    all_vars = parents + ([node] if node not in parents else [])
                    for var in all_vars:
                        for lag in range(1, self.lookback_periods + 1):
                            if len(predictions) < lag:
                                # Use historical data
                                val = historical_data[var].iloc[-(lag)]
                            else:
                                # Use previously predicted values
                                val = predictions[-lag][var]
                            features[f'{var}_lag_{lag}'] = val
                    
                    # Create feature DataFrame with correct column order
                    X = pd.DataFrame([features])
                    # Ensure feature columns match training data
                    X = X[self.models[node].feature_names_in_]
                    
                    pred_row[node] = float(self.models[node].predict(X)[0])
            
            predictions.append(pred_row)
        
        return pd.DataFrame(predictions)
    
    def run_counterfactual(self, interventions: Dict[str, float], steps: int = None) -> pd.DataFrame:
        """
        Run a counterfactual scenario.
        
        Args:
            interventions: Dictionary of node names and their intervention values
            steps: Number of steps to forecast
            
        Returns:
            DataFrame with counterfactual predictions
        """
        return self.predict(steps=steps, counterfactuals=interventions) 