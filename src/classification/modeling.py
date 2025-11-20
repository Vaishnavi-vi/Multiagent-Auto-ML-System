from src.state import MLState
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def classification__modeling_agent(state: MLState):
    preprocessor = state["preprocess"]
    
    model=LogisticRegression()

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipeline.fit(state["X_train"], state["y_train"])

    return {
        "model": pipeline
    }