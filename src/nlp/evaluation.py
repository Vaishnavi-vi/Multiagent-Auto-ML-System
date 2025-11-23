from src.state import MLState
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def nlp_evaluation_agent(state: MLState):
    model = state["best_model"]
    x_test = state["x_test"]
    y_test = state["y_test"]

    # Predict
    y_pred = model.predict(x_test)

    # --- FIX: handle numpy array y_test ---
    if isinstance(y_test, pd.Series):
        y_pred_series = pd.Series(y_pred, index=y_test.index)
    else:
        # y_test is numpy array → create default index
        y_pred_series = pd.Series(y_pred)

    # Metrics (universal for binary/multiclass)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    metrics = {
        "accuracy_score": accuracy_score(y_test, y_pred),
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1
    }

    return {
        "metrics": metrics,
        "y_pred": y_pred_series
    }
