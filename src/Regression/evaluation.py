from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from src.state import MLState
import pandas as pd

def regression_evaluation_agent(state: MLState):
    model = state["best_model"]
    X_test = state.get("X_test")
    y_test = state.get("y_test")

    y_pred= model.predict(X_test)
    y_pred_series=pd.Series(y_pred, index=y_test.index)
    metrics = {
            "mse": mean_squared_error(y_test,y_pred),
            "r2": r2_score(y_test,y_pred),
            "mae":mean_absolute_error(y_test,y_pred)
        }

    return {
        "metrics": metrics,
        "y_pred":y_pred_series
    }