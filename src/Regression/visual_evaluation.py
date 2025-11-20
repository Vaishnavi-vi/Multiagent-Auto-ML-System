from src.state import MLState
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

def regression_model_visualization_agent(state):
    y_true = state["y_test"]
    y_pred = state["y_pred"]
    
    print(f"y_true shape: {len(y_true)}, y_pred shape: {len(y_pred)}")  # DEBUG
    
    # Align if necessary
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length!")
    
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Regression: Actual vs Predicted")
    plt.show()
    
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    state["metrics"] = {"mse": mse, "r2": r2}
    print(f"MSE: {mse:.4f}, R2: {r2:.4f}")
    
    return state