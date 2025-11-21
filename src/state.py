from typing import TypedDict,Optional,Dict,Any
import pandas as pd



class MLState(TypedDict):
    csv_path: str
    target: Optional[str]
    df: Optional[pd.DataFrame]
    problem_type: Optional[str]
    cleaned: Optional[bool]
    x: Optional[pd.DataFrame]
    y: Optional[pd.Series]
    vectorizer:Optional[object]
    features:Optional[Any]
    X_train: Optional[pd.DataFrame]
    X_test: Optional[pd.DataFrame]
    y_train: Optional[pd.Series]
    y_test: Optional[pd.Series]
    y_pred:Optional[pd.Series]
    preprocess:Optional[object]
    results: Dict[str, Dict[str, Any]]   
    best_model_name: Optional[str]
    best_model: Optional[object]
    metrics: Optional[dict]
    visualization_done: Optional[bool]
    summary:str