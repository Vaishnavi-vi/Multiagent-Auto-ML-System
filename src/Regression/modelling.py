from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from src.state import MLState

def regression_modeling_agent(state: MLState):
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error

    x_train = state["x_train"]
    x_test = state["x_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]
    preprocess = state["preprocess"]

    models = {
        "Linear Regression": LinearRegression(),
        "Lasso Regression": Lasso(),
        "Ridge Regression": Ridge(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
    }

    results = {}

    for name, model in models.items():
        # Pipeline: Preprocessing + Model
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", model)
        ])

        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae=mean_absolute_error(y_test,y_pred)

        results[name] = {
            "r2_score": r2,
            "mse": mse,"mae":mae
        }

    # Pick best model by R²
        if "best_model" not in state or r2 > state.get("best_score", 0):
            state["best_model"] = pipe
            state["best_model_name"] = name
            state["best_score"] = r2
    
    
    state["results"]=results
    state["metrics"] = {name: res["r2_score"] for name, res in results.items()}


    return state
    

