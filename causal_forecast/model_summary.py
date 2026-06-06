import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from .utils import FittedNodeModel
from .typing import VariableType


def _glm_family_name(fitted: FittedNodeModel) -> str:
    model = fitted.model
    if hasattr(model, "family") and model.family is not None:
        return str(model.family.__class__.__name__)
    if fitted.variable_type == "binary":
        return "Binomial"
    if fitted.variable_type == "multiclass":
        return "Multinomial"
    if fitted.variable_type == "ordinal":
        return "Ordered"
    return "GLM"


def summarize_random_forest(fitted: FittedNodeModel) -> pd.DataFrame:
    """Feature importance table for a fitted Random Forest node model."""
    feature_names = fitted.feature_names or list(fitted.model.feature_names_in_)
    importances = fitted.model.feature_importances_

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def summarize_glm(fitted: FittedNodeModel) -> pd.DataFrame:
    """Coefficient table for a fitted GLM / MNLogit / OrderedModel node."""
    result = fitted.model
    params = result.params

    if isinstance(params, pd.Series):
        coef_df = params.rename("coef").to_frame()
    else:
        coef_df = pd.DataFrame(params)
        if coef_df.shape[1] == 1:
            coef_df.columns = ["coef"]
        else:
            coef_df = coef_df.stack().rename("coef").reset_index()
            coef_df.columns = ["equation", "feature", "coef"]

    if "feature" not in coef_df.columns:
        coef_df = coef_df.reset_index().rename(columns={"index": "feature"})

    for attr, col in (("bse", "std_err"), ("pvalues", "p_value"), ("tvalues", "stat")):
        if hasattr(result, attr):
            values = getattr(result, attr)
            if isinstance(values, pd.Series) and len(values) == len(coef_df):
                coef_df[col] = values.values
            elif isinstance(values, pd.DataFrame) and values.size == coef_df["coef"].size:
                coef_df[col] = values.values.flatten()

    if hasattr(result, "conf_int"):
        try:
            ci = result.conf_int()
            if isinstance(ci, pd.DataFrame) and len(ci) == len(coef_df):
                coef_df["ci_lower"] = ci.iloc[:, 0].values
                coef_df["ci_upper"] = ci.iloc[:, 1].values
        except Exception:
            pass

    sort_col = "p_value" if "p_value" in coef_df.columns else "coef"
    return coef_df.sort_values(sort_col, key=lambda s: s.abs(), ascending=False).reset_index(
        drop=True
    )


def summarize_node_model(fitted: FittedNodeModel, node: str) -> pd.DataFrame:
    """
    Detailed summary for a single fitted node model.

    Random Forest: feature importances.
    GLM: coefficients with standard errors and p-values when available.
    """
    if fitted.model_backend == "glm":
        detail = summarize_glm(fitted)
    else:
        detail = summarize_random_forest(fitted)

    detail.insert(0, "node", node)
    detail.insert(1, "variable_type", fitted.variable_type)
    detail.insert(2, "model_backend", fitted.model_backend)
    return detail


def summarize_node_overview(
    node: str,
    fitted: FittedNodeModel,
    variable_type: VariableType,
) -> Dict[str, object]:
    """One-row overview stats for a fitted node model."""
    detail = summarize_node_model(fitted, node)
    top = detail.iloc[0]

    overview = {
        "node": node,
        "variable_type": variable_type,
        "model_backend": fitted.model_backend,
        "n_features": len(fitted.feature_names or []),
    }

    if fitted.model_backend == "glm":
        overview["family"] = _glm_family_name(fitted)
        overview["top_feature"] = top.get("feature", None)
        overview["top_coef"] = float(top.get("coef", np.nan))
        if "p_value" in detail.columns:
            overview["top_p_value"] = float(top.get("p_value", np.nan))
        if hasattr(fitted.model, "llf"):
            overview["log_likelihood"] = float(fitted.model.llf)
        if hasattr(fitted.model, "aic"):
            overview["aic"] = float(fitted.model.aic)
    else:
        overview["family"] = None
        overview["top_feature"] = top.get("feature", None)
        overview["top_importance"] = float(top.get("importance", np.nan))
        overview["n_estimators"] = getattr(fitted.model, "n_estimators", None)

    return overview


def summarize_models(
    models: Dict[str, FittedNodeModel],
    variable_types: Dict[str, VariableType],
    node_order: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Overview summary for all fitted node models.

    Returns a DataFrame indexed by node with model type, backend, and top driver feature.
    """
    if not models:
        raise ValueError("No fitted models to summarize. Call fit() first.")

    order = node_order or list(models.keys())
    rows = [
        summarize_node_overview(node, models[node], variable_types[node])
        for node in order
        if node in models
    ]
    return pd.DataFrame(rows).set_index("node")


def summarize_models_detailed(
    models: Dict[str, FittedNodeModel],
    node_order: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Detailed per-node summaries keyed by graph node name."""
    if not models:
        raise ValueError("No fitted models to summarize. Call fit() first.")

    order = node_order or list(models.keys())
    return {node: summarize_node_model(models[node], node) for node in order if node in models}
