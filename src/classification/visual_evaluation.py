from src.state import MLState
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import streamlit as st  # <--- Streamlit

import io, base64

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_b64

def classification_model_visualization_agent(state: MLState):

    y_true = np.array(state.get("y_test"))
    y_pred = np.array(state.get("y_pred"))

    if y_true is None or y_pred is None:
        raise ValueError("y_test and y_pred must be present in state for visualization.")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    # Convert fig → base64
    img_b64 = fig_to_base64(fig)

    plt.close(fig)

    # Store in state
    if "plots" not in state:
        state["plots"] = {}

    state["plots"]["confusion_matrix"] = img_b64

    return state


