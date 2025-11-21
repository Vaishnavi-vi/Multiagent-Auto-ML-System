from src.state import MLState

def nlp_train_test_split(state: MLState):
    from sklearn.model_selection import train_test_split

    X = state["features"]
    y = state["df"][state["target"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return {
        **state,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
