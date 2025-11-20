from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from src.state import MLState

def regression_modeling_agent(state: MLState):
    from sklearn.linear_model import LinearRegression, Lasso, Ridge
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error

    x_train = state["X_train"]
    x_test = state["X_test"]
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

        results[name] = {
            "model": pipe,
            "r2_score": r2,
            "mse": mse
        }

    # Pick best model by R²
    best_model_name = max(results, key=lambda m: results[m]["r2_score"])
    best_model = results[best_model_name]["model"]

    return {
        "results": results,
        "best_model_name": best_model_name,
        "best_model": best_model
    }
