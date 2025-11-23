from src.state import MLState
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def classification_evaluation_agent(state: MLState):
    model = state["best_model"]
    x_test = state["x_test"]
    y_test = state["y_test"]
    
    y_pred = model.predict(x_test)
    y_pred_list = y_pred.tolist()  # JSON-safe

    metrics = {
        "accuracy_score": accuracy_score(y_test, y_pred),
        "precision_score": precision_score(y_test, y_pred, average='weighted'),
        "recall_score": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted')
    }

    state["metrics"] = metrics
    state["y_pred"] = y_pred_list  # JSON-safe

    return state

