import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
from typing import List, Optional, Union

from .typing import VariableType
from .utils import build_label_encoder, decode_values, encode_values


def _fitted_node_model():
    """Import at call time so reloads in notebooks pick up dataclass changes."""
    from .utils import FittedNodeModel

    return FittedNodeModel

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    HAS_ORDERED_MODEL = True
except ImportError:
    HAS_ORDERED_MODEL = False

ModelBackend = str  # "glm" or "random_forest"


def _as_float_design(X: pd.DataFrame) -> pd.DataFrame:
    return X.astype(float)


def _add_constant(X: pd.DataFrame) -> pd.DataFrame:
    return sm.add_constant(X, has_constant="add")


def _binary_target(y: pd.Series, label_encoder) -> np.ndarray:
    encoded = encode_values(y, label_encoder)
    return np.asarray(encoded, dtype=float)


def train_glm_node_model(
    X: pd.DataFrame,
    y: pd.Series,
    variable_type: VariableType,
    label_encoder=None,
):
    """Train a statsmodels GLM appropriate for the response variable type."""
    FittedNodeModel = _fitted_node_model()
    feature_names = list(X.columns)
    X_float = _as_float_design(X)
    X_const = _add_constant(X_float)

    if variable_type == "continuous":
        result = sm.GLM(y.astype(float), X_const, family=sm.families.Gaussian()).fit()
        return FittedNodeModel(
            model=result,
            variable_type=variable_type,
            model_backend="glm",
            feature_names=feature_names,
        )

    if label_encoder is None:
        label_encoder = build_label_encoder(y, variable_type)

    if variable_type == "binary":
        y_bin = _binary_target(y, label_encoder)
        result = sm.GLM(y_bin, X_const, family=sm.families.Binomial()).fit()
        return FittedNodeModel(
            model=result,
            variable_type=variable_type,
            label_encoder=label_encoder,
            categories=list(label_encoder.classes_),
            model_backend="glm",
            feature_names=feature_names,
        )

    y_encoded = encode_values(y, label_encoder).astype(int)

    if variable_type == "ordinal" and HAS_ORDERED_MODEL:
        try:
            result = OrderedModel(y_encoded, X_const, distr="logit").fit(
                method="bfgs",
                disp=0,
                maxiter=500,
            )
            return FittedNodeModel(
                model=result,
                variable_type=variable_type,
                label_encoder=label_encoder,
                categories=list(label_encoder.classes_),
                model_backend="glm",
                feature_names=feature_names,
            )
        except Exception:
            pass

    result = MNLogit(y_encoded, X_const).fit(disp=0, maxiter=500)
    return FittedNodeModel(
        model=result,
        variable_type=variable_type,
        label_encoder=label_encoder,
        categories=list(label_encoder.classes_),
        model_backend="glm",
        feature_names=feature_names,
    )


def _align_features(X: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    return _as_float_design(X[feature_names])


def predict_glm_value(
    fitted,
    X: pd.DataFrame,
    return_proba: bool = False,
) -> Union[float, int, str]:
    """Predict one row using a fitted GLM node model."""
    X_const = _add_constant(_align_features(X, fitted.feature_names))

    if fitted.variable_type == "continuous":
        return float(fitted.model.predict(X_const)[0])

    if fitted.variable_type == "binary":
        proba = float(fitted.model.predict(X_const)[0])
        if return_proba:
            return proba
        pred_encoded = int(proba >= 0.5)
    else:
        probs = fitted.model.predict(X_const)
        if probs.ndim == 1:
            pred_encoded = int(np.argmax(probs))
        else:
            pred_encoded = int(np.argmax(probs[0]))

    decoded = decode_values([pred_encoded], fitted.label_encoder, fitted.variable_type)
    value = decoded[0]
    if fitted.variable_type == "binary" and isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    return value


def predict_glm_batch(
    fitted,
    X: pd.DataFrame,
) -> np.ndarray:
    """Predict multiple rows using a fitted GLM node model."""
    X_const = _add_constant(_align_features(X, fitted.feature_names))

    if fitted.variable_type == "continuous":
        return np.asarray(fitted.model.predict(X_const), dtype=float)

    if fitted.variable_type == "binary":
        proba = np.asarray(fitted.model.predict(X_const))
        encoded = (proba >= 0.5).astype(int)
    else:
        probs = fitted.model.predict(X_const)
        if probs.ndim == 1:
            encoded = np.array([int(np.argmax(probs))])
        else:
            encoded = np.argmax(probs, axis=1).astype(int)

    return np.asarray(decode_values(encoded, fitted.label_encoder, fitted.variable_type))
