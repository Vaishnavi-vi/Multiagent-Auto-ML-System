from src.state import MLState

def nlp_modeling_agent(state: MLState):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import accuracy_score

    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "naive_bayes": MultinomialNB(),
    }

    best_model = None
    best_acc = -1
    best_name = None
    results = {}

    for name, model in models.items():
        model.fit(state["X_train"], state["y_train"])
        y_pred = model.predict(state["X_test"])
        acc = accuracy_score(state["y_test"], y_pred)

        # store result for comparison
        results[name] = {
            "accuracy": acc
        }

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    # merge with existing state
    return {
            "best_model": best_model,
            "best_model_name": best_name,
            "results": results,
            "metrics": {"accuracy": best_acc}}
    
    
    
    

