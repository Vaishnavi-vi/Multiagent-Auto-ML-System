from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from src.state import MLState
import pandas as pd

def regression_evaluation_agent(state: MLState):
    model = state["best_model"]
    x_test = state.get("x_test")
    y_test = state.get("y_test")

    y_pred= model.predict(x_test)
    y_pred_list = y_pred.tolist()
    metrics = {
            "mse": mean_squared_error(y_test,y_pred),
            "r2": r2_score(y_test,y_pred),
            "mae":mean_absolute_error(y_test,y_pred)
        }

    return {
        "metrics": metrics,
        "y_pred":y_pred_list
    }