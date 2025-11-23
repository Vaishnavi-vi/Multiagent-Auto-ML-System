from src.state import MLState
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import base64, io

def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return img_b64


def nlp_model_visualization_agent(state: MLState):

    y_true = state.get("y_test")
    y_pred = state.get("y_pred")

    if y_true is None or y_pred is None:
        raise ValueError("Missing y_test or y_pred in state.")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    # Convert to base64
    img_b64 = fig_to_base64(fig)

    # Save to state for FastAPI response
    state["plots"] = state.get("plots", {})
    state["plots"]["nlp_confusion_matrix"] = img_b64

    plt.close(fig)

    return state
