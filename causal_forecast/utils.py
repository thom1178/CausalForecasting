import networkx as nx
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from typing import Dict, List, Optional, Union

from .typing import VariableType


@dataclass
class FittedNodeModel:
    """Container for a per-node fitted model and its encoders."""

    model: Union[RandomForestRegressor, RandomForestClassifier]
    variable_type: VariableType
    label_encoder: Optional[LabelEncoder] = None
    categories: Optional[List] = field(default=None)
    one_hot_encoder: Optional[OneHotEncoder] = None
    one_hot_feature_names: Optional[List[str]] = field(default=None)


def validate_graph_data(data: pd.DataFrame, graph: nx.DiGraph, target: str):
    """Validate that the graph and data are compatible."""
    for node in graph.nodes():
        if node not in data.columns:
            raise ValueError(f"Node {node} from graph not found in data columns")

    if target not in data.columns:
        raise ValueError(f"Target variable {target} not found in data columns")

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be acyclic (DAG)")


def infer_time_delta(time_series: pd.Series) -> timedelta:
    """Infer the median timestep from a time column."""
    times = pd.to_datetime(time_series).sort_values()
    diffs = times.diff().dropna()
    if diffs.empty:
        return timedelta(days=1)
    median_delta = diffs.median()
    return median_delta.to_pytimedelta() if hasattr(median_delta, "to_pytimedelta") else timedelta(days=1)


def build_label_encoder(
    series: pd.Series,
    variable_type: VariableType,
) -> LabelEncoder:
    """Fit a label encoder for categorical target or lag features."""
    encoder = LabelEncoder()
    values = series.dropna()

    if variable_type == "ordinal" and isinstance(series.dtype, pd.CategoricalDtype):
        categories = list(series.dtype.categories)
        encoder.fit(categories)
    else:
        encoder.fit(values.astype(str))

    return encoder


def encode_values(
    values: Union[pd.Series, list, np.ndarray],
    encoder: LabelEncoder,
) -> np.ndarray:
    """Encode categorical values to numeric labels."""
    if isinstance(values, pd.Series):
        raw = values.astype(str)
    else:
        raw = pd.Series(values).astype(str)
    return encoder.transform(raw)


def decode_values(
    encoded: Union[np.ndarray, list, float, int],
    encoder: LabelEncoder,
    variable_type: VariableType,
) -> Union[np.ndarray, object]:
    """Decode model predictions back to original representation."""
    arr = np.asarray(encoded).reshape(-1)
    decoded = encoder.inverse_transform(arr.astype(int))

    if variable_type == "binary":
        unique = set(encoder.classes_)
        if unique.issubset({"0", "1"}):
            return decoded.astype(int)
        if unique.issubset({"0.0", "1.0"}):
            return decoded.astype(float)
        if unique.issubset({"True", "False", "true", "false"}):
            return np.array([v in ("True", "true", "1", 1) for v in decoded])

    return decoded


def build_one_hot_encoder(series: pd.Series) -> tuple:
    """Build a one-hot encoder for multiclass/ordinal parent lag features."""
    values = series.dropna().astype(str).values.reshape(-1, 1)
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoder.fit(values)
    feature_names = [
        f"cat_{cat}" for cat in encoder.categories_[0]
    ]
    return encoder, feature_names


def expand_categorical_lag_features(
    df: pd.DataFrame,
    var: str,
    lookback_periods: int,
    variable_type: VariableType,
    label_encoders: Dict[str, LabelEncoder],
    one_hot_encoders: Dict[str, OneHotEncoder],
    one_hot_feature_names: Dict[str, List[str]],
    use_one_hot: bool,
) -> List[str]:
    """
    Add lag features for a categorical variable.

    Phase 2: one-hot encode multiclass/ordinal parents for richer downstream features.
    """
    feature_cols: List[str] = []

    if use_one_hot and variable_type in ("multiclass", "ordinal"):
        if var not in one_hot_encoders:
            encoder, names = build_one_hot_encoder(df[var])
            one_hot_encoders[var] = encoder
            one_hot_feature_names[var] = names

        encoder = one_hot_encoders[var]
        names = one_hot_feature_names[var]

        for lag in range(1, lookback_periods + 1):
            lagged = df[var].shift(lag)
            missing = lagged.isna()
            lagged_str = lagged.astype(str).values.reshape(-1, 1)
            encoded = encoder.transform(lagged_str)
            for idx, name in enumerate(names):
                col = f"{var}_lag_{lag}_{name}"
                df[col] = encoded[:, idx]
                df.loc[missing, col] = np.nan
                feature_cols.append(col)
    else:
        if var not in label_encoders:
            label_encoders[var] = build_label_encoder(df[var], variable_type)

        encoded = encode_values(df[var], label_encoders[var])
        df[f"{var}_encoded"] = encoded

        for lag in range(1, lookback_periods + 1):
            col = f"{var}_lag_{lag}"
            df[col] = df[f"{var}_encoded"].shift(lag)
            feature_cols.append(col)

    return feature_cols


def train_node_model(
    X: pd.DataFrame,
    y: pd.Series,
    variable_type: VariableType,
    label_encoder: Optional[LabelEncoder] = None,
) -> FittedNodeModel:
    """Train a type-appropriate model for a single node."""
    if variable_type == "continuous":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return FittedNodeModel(model=model, variable_type=variable_type)

    if label_encoder is None:
        label_encoder = build_label_encoder(y, variable_type)

    y_encoded = encode_values(y, label_encoder)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y_encoded)

    return FittedNodeModel(
        model=model,
        variable_type=variable_type,
        label_encoder=label_encoder,
        categories=list(label_encoder.classes_),
    )


def predict_node_value(
    fitted: FittedNodeModel,
    X: pd.DataFrame,
    return_proba: bool = False,
) -> Union[float, int, str, np.ndarray]:
    """Predict a single row for a node using the fitted type-aware model."""
    if fitted.variable_type == "continuous":
        return float(fitted.model.predict(X)[0])

    pred_encoded = fitted.model.predict(X)[0]

    if return_proba and fitted.variable_type == "binary":
        proba = fitted.model.predict_proba(X)[0]
        return float(proba[1])

    decoded = decode_values([pred_encoded], fitted.label_encoder, fitted.variable_type)
    value = decoded[0]

    if fitted.variable_type == "binary" and isinstance(value, (np.bool_, bool)):
        return int(value)
    if fitted.variable_type == "binary" and isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    return value


def value_for_lag_feature(
    value: object,
    var: str,
    variable_type: VariableType,
    label_encoders: Dict[str, LabelEncoder],
) -> float:
    """Convert a predicted or historical value into a numeric lag feature."""
    if variable_type == "continuous":
        return float(value)

    if var not in label_encoders:
        raise KeyError(f"Label encoder for '{var}' not found.")

    encoded = encode_values([value], label_encoders[var])
    return float(encoded[0])
