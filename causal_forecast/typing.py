import networkx as nx
import pandas as pd
from typing import Dict, Literal, Optional, Set, Tuple

VariableType = Literal["continuous", "binary", "multiclass", "ordinal"]

BINARY_VALUE_SETS: Tuple[Set, ...] = (
    {0, 1},
    {0.0, 1.0},
    {True, False},
    {"0", "1"},
    {"yes", "no"},
    {"Yes", "No"},
    {"true", "false"},
    {"True", "False"},
)


def _normalize_unique(values: pd.Series) -> Set:
    return set(values.dropna().unique())


def _is_binary(series: pd.Series) -> bool:
    unique = _normalize_unique(series)
    if len(unique) != 2:
        return False
    normalized = set()
    for v in unique:
        if isinstance(v, str):
            normalized.add(v.strip().lower())
        elif isinstance(v, (bool,)):
            normalized.add(v)
        elif isinstance(v, (int, float)) and v in (0, 1):
            normalized.add(int(v))
        else:
            normalized.add(v)
    for pattern in BINARY_VALUE_SETS:
        if unique.issubset(pattern) or normalized.issubset(
            {x.lower() if isinstance(x, str) else x for x in pattern}
        ):
            return True
    return len(unique) == 2


def _is_ordered_categorical(series: pd.Series) -> bool:
    if isinstance(series.dtype, pd.CategoricalDtype):
        return series.dtype.ordered
    return False


def _detect_single_type(
    series: pd.Series,
    max_categories: int = 20,
) -> VariableType:
    if _is_ordered_categorical(series):
        return "ordinal"

    if pd.api.types.is_bool_dtype(series):
        return "binary"

    if pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        unique = series.dropna().unique()
        if len(unique) == 2:
            return "binary"
        return "multiclass"

    if _is_binary(series):
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        n_unique = series.dropna().nunique()
        if n_unique <= 2:
            return "binary"
        if n_unique <= max_categories and pd.api.types.is_integer_dtype(series):
            return "multiclass"
        return "continuous"

    return "multiclass"


def detect_variable_types(
    data: pd.DataFrame,
    graph: nx.DiGraph,
    time_column: str,
    overrides: Optional[Dict[str, VariableType]] = None,
    max_categories: int = 20,
) -> Dict[str, VariableType]:
    """
    Detect whether each graph node is continuous, binary, multiclass, or ordinal.

    Heuristics (lowest priority last):
    1. User overrides always win
    2. Ordered pandas CategoricalDtype -> ordinal
    3. Two unique values -> binary
    4. Object/category dtype -> multiclass (or binary if 2 values)
    5. Integer with few unique values -> multiclass
    6. Otherwise numeric -> continuous
    """
    overrides = overrides or {}
    variable_types: Dict[str, VariableType] = {}

    for node in graph.nodes():
        if node in overrides:
            variable_types[node] = overrides[node]
            continue
        variable_types[node] = _detect_single_type(data[node], max_categories=max_categories)

    return variable_types
