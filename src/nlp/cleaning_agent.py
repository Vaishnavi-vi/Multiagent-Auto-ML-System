from src.state import MLState
import pandas as pd
import re

def nlp_cleaning_agent(state: MLState):
    df = state["df"]
    target_col = state["target"]
    
    text_candidates = df.select_dtypes(include=["object"]).columns.tolist()
    text_candidates = [col for col in text_candidates if col != target_col]

    # Pick the text column with the highest average length
    text_col = max(text_candidates, key=lambda c: df[c].astype(str).str.len().mean())

    # 3. Perform cleaning on auto-detected column
    df["text_clean"] = (
        df[text_col]
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 4. Save detected text column for later stages
    state["text"] = text_col

    return {"df": df, "cleaned": True, "text": text_col}




