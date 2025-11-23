from typing import Optional, TypedDict,Dict,Any
import pandas as pd

class MLState(TypedDict):
    csv_path: Optional[str]               # optional, for in-memory CSV
    target: Optional[str]
    df: Optional[pd.DataFrame]
    problem_type: Optional[str]
    
    # cleaning/preprocessing
    cleaned: Optional[bool]
    numeric_features: Optional[list]
    categorical_features: Optional[list]
    missing_values: Optional[pd.DataFrame]
    basic_stats: Optional[pd.DataFrame]
    correlation: Optional[pd.DataFrame]
    distribution_plots_done: Optional[bool]
    correlation_plots_done: Optional[bool]
    outliers_detected: Optional[Dict[str, Any]]
    
    x: Optional[pd.DataFrame]
    y: Optional[pd.Series]
    vectorizer: Optional[object]
    features: Optional[Any]
    preprocess: Optional[object]
    
    # splits
    x_train: Optional[pd.DataFrame]
    x_test: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    y_pred: Optional[pd.Series]

    results: Dict[str, Dict[str, float]]        # only scores, JSON-serializable
    best_model_name: Optional[str]
    best_model: Optional[str]                   # model name only
    metrics: Optional[dict]
    visualization_done: Optional[bool]
    
    summary: Optional[str]                      # optional initially
    plots:  Dict[str, str]                      # base64 strings, JSON-serializable
