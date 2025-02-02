import networkx as nx
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from typing import List

def validate_graph_data(data: pd.DataFrame, graph: nx.DiGraph, target: str):
    """Validate that the graph and data are compatible."""
    # Check if all nodes in graph exist in data
    for node in graph.nodes():
        if node not in data.columns:
            raise ValueError(f"Node {node} from graph not found in data columns")
    
    # Check if target exists
    if target not in data.columns:
        raise ValueError(f"Target variable {target} not found in data columns")
    
    # Check for cycles
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph must be acyclic (DAG)")

def train_node_model(X: pd.DataFrame, y: pd.Series) -> RandomForestRegressor:
    """Train a model for a single node."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model 