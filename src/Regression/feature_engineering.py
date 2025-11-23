from src.state import MLState
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



def regression_feature_engineering_agent(state:MLState):
    df = state["df"]
    target = state["target"]

    x = df.drop(columns=[target])
    y = df[target]

    num_cols = x.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        )
    if cat_cols:
        transformers.append(
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)


    # Update state
    state["x"] = x
    state["y"] = y
    state["preprocess"] = preprocessor

    return state
