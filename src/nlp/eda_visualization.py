from src.state import MLState
import matplotlib.pyplot as plt
import pandas as pd
import io, base64


def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode("utf-8")
    buffer.close()
    return img_b64


def nlp_eda_agent(state: MLState):

    df = state["df"]
    target = state["target"]
    text_col = target 

    # Ensure plot storage exists
    if "plots" not in state:
        state["plots"] = {}


    fig1, ax1 = plt.subplots(figsize=(6, 6))
    df[target].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax1)
    ax1.set_title("Target Class Distribution")
    plt.tight_layout()

    state["plots"]["target_distribution"] = fig_to_base64(fig1)
    plt.close(fig1)

    return state




   
