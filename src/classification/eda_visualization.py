import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from src.state import MLState


# ------------------------------------------
# Convert Matplotlib Figure → Base64 string
# ------------------------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close(fig)  # Prevent memory leak
    return img_b64


# ------------------------------------------
# Main EDA Visualization Agent for Classification
# ------------------------------------------
def classification_eda_visualization_agent(state: MLState):
    df = state["df"]

    # Extract numerical + categorical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    state["plots"] = {}

    # ------------------------------------------
    # Numerical Feature Distributions
    # ------------------------------------------
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df[col], kde=True, alpha=0.5, ax=ax)
        ax.set_title(f"Distribution of {col}")
        plt.tight_layout()

        img_b64 = fig_to_base64(fig)
        state["plots"][f"{col}_hist"] = img_b64

    # ------------------------------------------
    # Categorical Feature Count Plots
    # ------------------------------------------
    for col in cat_cols:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=col, data=df, ax=ax)
        ax.set_title(f"Counts of {col}")
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        img_b64 = fig_to_base64(fig)
        state["plots"][f"{col}_count"] = img_b64

    # ------------------------------------------
    # Correlation Heatmap (only if >1 numerical col)
    # ------------------------------------------
    if len(num_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, ax=ax)
        ax.set_title("Correlation Heatmap of Numerical Features")
        plt.tight_layout()

        img_b64 = fig_to_base64(fig)
        state["plots"]["corr_heatmap"] = img_b64

    # Update state flags
    state["distribution_plots_done"] = True
    state["correlation_plots_done"] = True

    return state

