from src.state import MLState


def nlp_cleaning_agent(state: MLState):
    import re
    import pandas as pd

    df = state["df"]
    target = state["target"]

    df["text_clean"] = (
        df[target]
        .str.lower()
        .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    return {"df": df, "cleaned": True}
