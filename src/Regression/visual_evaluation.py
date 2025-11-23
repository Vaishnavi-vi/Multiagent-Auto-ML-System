from src.state import MLState
import matplotlib.pyplot as plt
import io, base64
import numpy as np

def regression_model_visualization_agent(state):
    y_true = np.array(state.get("y_test"))
    y_pred = np.array(state.get("y_pred"))

    # Debug shapes (optional)
    print(f"y_true shape: {len(y_true)}, y_pred shape: {len(y_pred)}")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length!")

    # Create Plot
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y_true, y_pred, alpha=0.6)
    ax.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        'r--',
        lw=2
    )

    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Regression: Actual vs Predicted")
    plt.tight_layout()

    # Convert figure -> Base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # Save inside state
    state["plots"] = state.get("plots", {})
    state["plots"]["regression_actual_vs_pred"] = img_b64

    plt.close(fig)

    return state

