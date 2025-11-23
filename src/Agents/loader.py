from src.state import MLState

def loader_agent(state: MLState):
    """
    Load CSV into state['df'].
    Accepts either:
    - csv_path (string)
    - df already provided (from Streamlit/FastAPI)
    """
    import pandas as pd

    # 1. If df is already provided, use it
    if state.get("df") is not None:
        state["csv_path"] = None
        return state

    # 2. Otherwise, read from csv_path
    path = state.get("csv_path")
    if not path:
        raise ValueError("No CSV path or DataFrame provided.")

    df = pd.read_csv(path)
    state["df"] = df
    return state

