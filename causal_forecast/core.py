import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime

from .typing import VariableType, detect_variable_types
from .utils import (
    FittedNodeModel,
    ModelType,
    infer_time_delta,
    train_node_model,
    validate_graph_data,
    build_label_encoder,
    expand_categorical_lag_features,
    predict_node_batch,
    predict_node_value,
    value_for_lag_feature,
)
from .metrics import (
    evaluate_by_horizon,
    evaluate_forecast_typed,
    summarize_backtest,
)
from .seasonality import (
    add_cyclical_time_features,
    add_seasonal_lag_features,
    cyclical_features_from_date,
    infer_data_frequency,
    validate_seasonality,
)
from .model_summary import (
    summarize_models,
    summarize_models_detailed,
    summarize_node_model,
)

CounterfactualValue = Union[float, int, str, List[Union[float, int, str]]]
CounterfactualSpec = Optional[Dict[str, CounterfactualValue]]


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
        lookback_periods: int = 3,
        variable_types: Optional[Dict[str, VariableType]] = None,
        type_overrides: Optional[Dict[str, VariableType]] = None,
        use_one_hot_parents: bool = True,
        seasonality: Optional[List[str]] = None,
        model_type: ModelType = "random_forest",
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
            variable_types: Explicit variable types per node (skips auto-detection)
            type_overrides: Overrides applied on top of auto-detection
            use_one_hot_parents: One-hot encode multiclass/ordinal parent lags (Phase 2)
            seasonality: Opt-in seasonal patterns ('weekly', 'monthly', 'yearly')
            model_type: Per-node model backend ('random_forest' or 'glm')
        """
        self.data = data.copy()
        self.graph = graph.copy()
        self.target = target
        self.time_column = time_column
        self.forecast_horizon = forecast_horizon
        self.lookback_periods = lookback_periods
        self.use_one_hot_parents = use_one_hot_parents
        self.seasonality = seasonality or []
        self.model_type = model_type

        validate_graph_data(self.data, self.graph, self.target)
        self.data = self.data.sort_values(time_column).reset_index(drop=True)

        if variable_types is not None:
            self.variable_types = variable_types.copy()
        else:
            self.variable_types = detect_variable_types(
                self.data,
                self.graph,
                self.time_column,
                overrides=type_overrides,
            )

        self.models: Dict[str, FittedNodeModel] = {}
        self.label_encoders: Dict[str, object] = {}
        self.one_hot_encoders: Dict[str, object] = {}
        self.one_hot_feature_names: Dict[str, List[str]] = {}
        self.time_delta = infer_time_delta(self.data[self.time_column])
        self.data_frequency = infer_data_frequency(self.time_delta)
        self.seasonal_periods = validate_seasonality(
            len(self.data),
            self.lookback_periods,
            self.seasonality,
            self.data_frequency,
        )
        seasonal_values = list(self.seasonal_periods.values())
        self.max_history = max([self.lookback_periods] + seasonal_values)
        self.cyclical_features: List[str] = []
        self.node_order = list(nx.topological_sort(self.graph))

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from the time column."""
        df = df.copy()
        dt = pd.to_datetime(df[self.time_column])
        df["year"] = dt.dt.year
        df["month"] = dt.dt.month
        df["day"] = dt.dt.day
        df["dayofweek"] = dt.dt.dayofweek

        if self.seasonal_periods:
            df, self.cyclical_features = add_cyclical_time_features(df, self.time_column)

        return df

    def _prepare_time_series_data(self, node: str, return_dates: bool = False) -> tuple:
        """Prepare time series data with lagged features."""
        df = self._create_time_features(self.data)
        parents = list(self.graph.predecessors(node))
        features: List[str] = []

        time_features = ["year", "month", "day", "dayofweek"]
        if self.cyclical_features:
            time_features.extend(self.cyclical_features)
        features.extend(time_features)

        all_vars = parents + ([node] if node not in parents else [])
        for var in all_vars:
            var_type = self.variable_types[var]
            if var_type == "continuous":
                for lag in range(1, self.lookback_periods + 1):
                    col = f"{var}_lag_{lag}"
                    df[col] = df[var].shift(lag)
                    features.append(col)
            else:
                lag_cols = expand_categorical_lag_features(
                    df,
                    var,
                    self.lookback_periods,
                    var_type,
                    self.label_encoders,
                    self.one_hot_encoders,
                    self.one_hot_feature_names,
                    use_one_hot=self.use_one_hot_parents,
                )
                features.extend(lag_cols)

            if self.seasonal_periods:
                seasonal_cols = add_seasonal_lag_features(
                    df,
                    var,
                    self.seasonal_periods,
                    var_type,
                    self.label_encoders,
                    self.one_hot_encoders,
                    self.one_hot_feature_names,
                    use_one_hot=self.use_one_hot_parents,
                )
                features.extend(seasonal_cols)

        # Drop rows without full lag history so all nodes share the same time index.
        df = df.iloc[self.max_history :].dropna()
        X = df[features]
        y = df[node]

        if return_dates:
            return X, y, df[self.time_column].reset_index(drop=True)
        return X, y

    def fit(self, verbose: bool = True):
        """Train type-aware models for each node in the causal graph."""
        for node in self.node_order:
            if verbose:
                print(
                    f"Training model for node: {node} "
                    f"({self.variable_types[node]}, {self.model_type})"
                )

            X, y = self._prepare_time_series_data(node)
            var_type = self.variable_types[node]

            label_encoder = None
            if var_type != "continuous":
                label_encoder = build_label_encoder(y, var_type)

            self.models[node] = train_node_model(
                X,
                y,
                var_type,
                label_encoder=label_encoder,
                model_type=self.model_type,
            )

    def summarize_models(self, detailed: bool = False) -> Union[pd.DataFrame, tuple]:
        """
        Summarize each fitted per-node model.

        Args:
            detailed: If True, return (overview_df, details_dict) where details_dict
                maps node names to coefficient/importance tables.

        Returns:
            Overview DataFrame indexed by node, or tuple with detailed tables.
        """
        if not self.models:
            raise ValueError("No fitted models to summarize. Call fit() first.")

        overview = summarize_models(
            self.models,
            self.variable_types,
            node_order=self.node_order,
        )
        if detailed:
            return overview, summarize_models_detailed(self.models, node_order=self.node_order)
        return overview

    def summarize_model(self, node: str) -> pd.DataFrame:
        """Detailed summary for a single node's fitted model."""
        if node not in self.models:
            raise ValueError(f"No fitted model for node '{node}'. Available: {list(self.models)}")
        return summarize_node_model(self.models[node], node)

    def _future_dates(self, steps: int) -> List[datetime]:
        last_date = pd.to_datetime(self.data[self.time_column].max())
        return [last_date + self.time_delta * (i + 1) for i in range(steps)]

    @staticmethod
    def _resolve_counterfactual(
        counterfactuals: CounterfactualSpec,
        node: str,
        step_idx: int,
        steps: int,
    ) -> Optional[object]:
        if not counterfactuals or node not in counterfactuals:
            return None

        value = counterfactuals[node]
        if isinstance(value, list):
            if len(value) != steps:
                raise ValueError(
                    f"Counterfactual for '{node}' has length {len(value)}, expected {steps}."
                )
            return value[step_idx]
        return value

    def _lag_raw_value(
        self,
        var: str,
        lag: int,
        predictions: List[Dict],
        historical_data: pd.DataFrame,
    ) -> object:
        if len(predictions) >= lag:
            return predictions[-lag][var]
        offset = lag - len(predictions)
        return historical_data[var].iloc[-offset]

    def _lag_value(
        self,
        var: str,
        lag: int,
        predictions: List[Dict],
        historical_data: pd.DataFrame,
    ) -> float:
        raw = self._lag_raw_value(var, lag, predictions, historical_data)
        return value_for_lag_feature(
            raw,
            var,
            self.variable_types[var],
            self.label_encoders,
        )

    def _build_prediction_features(
        self,
        node: str,
        future_date: datetime,
        predictions: List[Dict],
        historical_data: pd.DataFrame,
    ) -> pd.DataFrame:
        parents = list(self.graph.predecessors(node))
        time_features = {
            "year": future_date.year,
            "month": future_date.month,
            "day": future_date.day,
            "dayofweek": future_date.dayofweek,
        }
        if self.seasonal_periods:
            time_features.update(cyclical_features_from_date(future_date))
        features = dict(time_features)

        all_vars = parents + ([node] if node not in parents else [])
        for var in all_vars:
            var_type = self.variable_types[var]
            if var_type == "continuous":
                for lag in range(1, self.lookback_periods + 1):
                    features[f"{var}_lag_{lag}"] = self._lag_value(var, lag, predictions, historical_data)
            elif self.use_one_hot_parents and var_type in ("multiclass", "ordinal"):
                encoder = self.one_hot_encoders[var]
                names = self.one_hot_feature_names[var]
                for lag in range(1, self.lookback_periods + 1):
                    hist_raw = self._lag_raw_value(var, lag, predictions, historical_data)
                    vec = encoder.transform([[str(hist_raw)]])[0]
                    for idx, name in enumerate(names):
                        features[f"{var}_lag_{lag}_{name}"] = float(vec[idx])
            else:
                for lag in range(1, self.lookback_periods + 1):
                    features[f"{var}_lag_{lag}"] = self._lag_value(var, lag, predictions, historical_data)

            if self.seasonal_periods:
                for season_name, period in self.seasonal_periods.items():
                    col = f"{var}_s_lag_{season_name}"
                    if var_type == "continuous":
                        features[col] = self._lag_value(var, period, predictions, historical_data)
                    elif self.use_one_hot_parents and var_type in ("multiclass", "ordinal"):
                        hist_raw = self._lag_raw_value(var, period, predictions, historical_data)
                        vec = self.one_hot_encoders[var].transform([[str(hist_raw)]])[0]
                        for idx, name in enumerate(self.one_hot_feature_names[var]):
                            features[f"{col}_{name}"] = float(vec[idx])
                    else:
                        features[col] = self._lag_value(var, period, predictions, historical_data)

        X = pd.DataFrame([features])
        fitted = self.models[node]
        feature_names = fitted.feature_names or list(fitted.model.feature_names_in_)
        return X[feature_names]

    def predict(
        self,
        steps: int = None,
        counterfactuals: CounterfactualSpec = None,
        return_proba: bool = False,
    ) -> pd.DataFrame:
        """
        Make time series predictions using the causal model.

        Args:
            steps: Forecast horizon
            counterfactuals: Per-node intervention values. Scalars apply to all steps;
                lists must have length equal to steps.
            return_proba: For binary nodes, return probability instead of class label
        """
        steps = steps or self.forecast_horizon
        predictions: List[Dict] = []
        future_dates = self._future_dates(steps)
        historical_data = self.data.copy().tail(self.max_history)

        for step_idx, future_date in enumerate(future_dates):
            pred_row = {self.time_column: future_date}

            for node in self.node_order:
                intervention = self._resolve_counterfactual(counterfactuals, node, step_idx, steps)
                if intervention is not None:
                    pred_row[node] = intervention
                else:
                    X = self._build_prediction_features(node, future_date, predictions, historical_data)
                    pred_row[node] = predict_node_value(
                        self.models[node],
                        X,
                        return_proba=return_proba and self.variable_types[node] == "binary",
                    )

            predictions.append(pred_row)

        return pd.DataFrame(predictions)

    def predict_in_sample(self) -> pd.DataFrame:
        """Generate one-step-ahead in-sample predictions for all nodes."""
        dates = None
        predictions: Dict[str, np.ndarray] = {}

        for node in self.node_order:
            if node not in self.models:
                raise ValueError("Models not fitted. Call fit() before predict_in_sample().")

            X, _, node_dates = self._prepare_time_series_data(node, return_dates=True)
            predictions[node] = predict_node_batch(self.models[node], X)

            if dates is None:
                dates = node_dates
            elif not dates.equals(node_dates):
                raise ValueError("Inconsistent date alignment across nodes during in-sample prediction.")

        result = pd.DataFrame(predictions)
        result[self.time_column] = dates.values
        return result[[self.time_column] + self.node_order]

    def evaluate(
        self,
        holdout_steps: int = None,
        variables: Optional[List[str]] = None,
        in_sample: bool = False,
        return_predictions: bool = False,
    ) -> Union[pd.DataFrame, tuple]:
        """Evaluate forecast accuracy with type-aware metrics."""
        variables = variables or list(self.graph.nodes())

        if in_sample:
            predictions = self.predict_in_sample()
            actual = self.data
        else:
            holdout_steps = holdout_steps or self.forecast_horizon
            if holdout_steps >= len(self.data):
                raise ValueError(
                    f"holdout_steps ({holdout_steps}) must be less than data length ({len(self.data)})."
                )

            train_data = self.data.iloc[:-holdout_steps]
            test_data = self.data.iloc[-holdout_steps:]

            eval_forecaster = self._clone_with_data(train_data)
            eval_forecaster.fit(verbose=False)
            predictions = eval_forecaster.predict(steps=holdout_steps)
            actual = test_data

        metrics = evaluate_forecast_typed(
            actual,
            predictions,
            self.time_column,
            variables,
            self.variable_types,
        )
        if return_predictions:
            return metrics, predictions, actual
        return metrics

    def evaluate_horizon(
        self,
        variable: str = None,
        holdout_steps: int = None,
    ) -> pd.DataFrame:
        """Evaluate forecast error at each horizon step for a single variable."""
        variable = variable or self.target
        holdout_steps = holdout_steps or self.forecast_horizon

        if holdout_steps >= len(self.data):
            raise ValueError(
                f"holdout_steps ({holdout_steps}) must be less than data length ({len(self.data)})."
            )

        train_data = self.data.iloc[:-holdout_steps]
        test_data = self.data.iloc[-holdout_steps:]

        eval_forecaster = self._clone_with_data(train_data)
        eval_forecaster.fit(verbose=False)
        predictions = eval_forecaster.predict(steps=holdout_steps)

        return evaluate_by_horizon(test_data, predictions, self.time_column, variable)

    def backtest(
        self,
        horizon: int = None,
        min_train_size: int = None,
        step_size: int = 1,
        variables: Optional[List[str]] = None,
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """
        Walk-forward backtest with expanding training window.

        Each fold retrains only on data strictly before the test window to prevent leakage.
        """
        horizon = horizon or self.forecast_horizon
        variables = variables or list(self.graph.nodes())
        min_train_size = min_train_size or max(self.lookback_periods * 3, 10)

        if min_train_size + horizon > len(self.data):
            raise ValueError(
                "Not enough data for backtest. "
                f"Need at least min_train_size + horizon ({min_train_size + horizon}) rows."
            )

        cutoffs = range(min_train_size, len(self.data) - horizon + 1, step_size)

        if n_jobs == 1:
            records = [self._run_backtest_fold(cutoff, horizon, variables) for cutoff in cutoffs]
        else:
            from joblib import Parallel, delayed

            records = Parallel(n_jobs=n_jobs)(
                delayed(self._run_backtest_fold)(cutoff, horizon, variables) for cutoff in cutoffs
            )

        return pd.concat(records, ignore_index=True)

    def summarize_backtest(
        self,
        backtest_df: pd.DataFrame,
        metric: str = "rmse",
    ) -> pd.DataFrame:
        """Aggregate a backtest result DataFrame."""
        return summarize_backtest(backtest_df, metric=metric, variable_types=self.variable_types)

    def _run_backtest_fold(
        self,
        cutoff: int,
        horizon: int,
        variables: List[str],
    ) -> pd.DataFrame:
        train_data = self.data.iloc[:cutoff]
        test_data = self.data.iloc[cutoff : cutoff + horizon]

        fold_forecaster = self._clone_with_data(train_data)
        fold_forecaster.fit(verbose=False)
        predictions = fold_forecaster.predict(steps=horizon)

        actual_time = self.time_column
        pred_time = self.time_column if self.time_column in predictions.columns else "timestamp"

        rows = []
        for step_idx in range(horizon):
            for var in variables:
                actual_val = test_data.iloc[step_idx][var]
                pred_val = predictions.iloc[step_idx][var]
                rows.append(
                    {
                        "fold": cutoff,
                        "train_end": train_data[self.time_column].iloc[-1],
                        "horizon_step": step_idx + 1,
                        "timestamp": test_data.iloc[step_idx][actual_time],
                        "variable": var,
                        "variable_type": self.variable_types[var],
                        "actual": actual_val,
                        "predicted": pred_val,
                    }
                )

        return pd.DataFrame(rows)

    def _clone_with_data(self, data: pd.DataFrame) -> "CausalForecaster":
        return CausalForecaster(
            data=data,
            graph=self.graph,
            target=self.target,
            time_column=self.time_column,
            forecast_horizon=self.forecast_horizon,
            lookback_periods=self.lookback_periods,
            variable_types=self.variable_types,
            use_one_hot_parents=self.use_one_hot_parents,
            seasonality=self.seasonality,
            model_type=self.model_type,
        )

    def run_counterfactual(
        self,
        interventions: CounterfactualSpec,
        steps: int = None,
    ) -> pd.DataFrame:
        """
        Run a counterfactual scenario.

        Args:
            interventions: Per-node values (scalar or list with length == steps)
            steps: Number of steps to forecast
        """
        return self.predict(steps=steps, counterfactuals=interventions)
