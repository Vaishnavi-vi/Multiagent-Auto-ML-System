from src.state import MLState


from sklearn.model_selection import train_test_split

def classification_train_test_split_agent(state: MLState):
    x, y = state["x"], state["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }