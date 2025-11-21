from src.state import MLState
import pandas as pd

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def nlp_evaluation_agent(state: MLState):
    model = state["best_model"]
    X_test = state.get("X_test")
    y_test = state.get("y_test")
    
    
    y_pred= model.predict(X_test)
    y_pred_series=pd.Series(y_pred, index=y_test.index)
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    metrics = {
        "accuracy_score":accuracy_score(y_test,y_pred),
        "precision_score":precision,
        "recall_score":recall,
        "f1_score":f1 
        }

    return {
        "metrics": metrics,
        "y_pred":y_pred_series
    }