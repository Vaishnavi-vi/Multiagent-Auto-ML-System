from src.state import MLState
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def regression_feature_engineering_agent(state: MLState):
    df = state["df"]
    target = state["target"]

    X = df.drop(columns=[target])
    y = df[target]

    # identify columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Build Transformer Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            # Numeric: Imputer → Scaler
            ("num", 
             Pipeline([
                 ("imputer", SimpleImputer(strategy="median")),
                 ("scaler", StandardScaler())
             ]), 
             num_cols),

            # Categorical: Imputer → OneHotEncoder
            ("cat", 
             Pipeline([
                 ("imputer", SimpleImputer(strategy="most_frequent")),
                 ("encoder", OneHotEncoder(handle_unknown="ignore"))
             ]), 
             cat_cols),
        ]
    )

    return {
        "preprocess": preprocessor,
        "x": X,
        "y": y
    }
