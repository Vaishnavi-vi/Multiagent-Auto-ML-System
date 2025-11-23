from src.state import MLState
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def classification_modeling_agent(state: MLState):
    x_train = state["x_train"]
    x_test = state["x_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]
    preprocess = state["preprocess"]

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }

    results = {}

    for name, model in models.items():
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", model)
        ])

        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        pre=precision_score(y_test,y_pred,average="weighted")
        recall=recall_score(y_test,y_pred,average="weighted")
        f1=f1_score(y_test,y_pred,average="weighted")

        # Only store metrics
        results[name] = {
            "accuracy_score": accuracy,
            "precision_score":pre,
            "recall_score":recall,
            "f1_score":f1
        }

        # Optionally store best pipeline internally
        if "best_model" not in state or accuracy > state.get("best_score", 0):
            state["best_model"] = pipe
            state["best_model_name"] = name
            state["best_score"] = accuracy

    state["results"] = results
    state["metrics"] = {name: res["accuracy_score"] for name, res in results.items()}

    return state
