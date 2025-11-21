from src.state import MLState
import pandas as pd


def problem_detector_agent(state: MLState):
    df = state["df"]
    target = state["target"]
    y = df[target]


    if pd.api.types.is_string_dtype(y):
        problem_type = "nlp"

    elif pd.api.types.is_numeric_dtype(y):
        if y.nunique() <= 20 and y.nunique() < len(y) * 0.05:
            problem_type = "classification"
        else:
            problem_type = "regression"
            
    else:
        problem_type = "classification"

    return {
        **state,
        "problem_type": problem_type
    }
