from src.state import MLState


def classification_cleaning_agent(state: MLState):
    df = state["df"]
    target = state["target"]

    numeric_features = df.select_dtypes(include="number").columns.to_list()
    categorical_features = df.select_dtypes(exclude="number").columns.to_list()
    
    text_cols = df.select_dtypes(include=['object']).columns
    text_cols = [col for col in text_cols if col != target]  # keep target if needed
    df = df.drop(columns=text_cols)

    # Stats
    missing_values = df.isnull().sum().to_dict()
    basic_stats = df.describe().to_dict()
    correlation = df.corr().to_dict()
    

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    state["target"] = state["target"].strip().lower().replace(" ", "_")

    # Drop fully null columns
    df = df.dropna(axis=1, how="all")
    # Drop rows where target is null
    df = df[df[state["target"]].notna()].reset_index(drop=True)
    # Drop duplicates
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    # Drop constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    # Drop columns with >80% nulls
    df = df.loc[:, df.isnull().mean() < 0.8]

    # Outlier clipping
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        skew = df[col].skew()
        low_q, high_q = (0.05, 0.95) if abs(skew) < 1 else (0.01, 0.99)
        lower, upper = df[col].quantile(low_q), df[col].quantile(high_q)
        df[col] = df[col].clip(lower, upper)

    # Update state
    state["df"] = df
    state["cleaned"] = True
    state["numeric_features"] = numeric_features
    state["categorical_features"] = categorical_features
    state["missing_values"] = missing_values
    state["basic_stats"] = basic_stats
    state["correlation"] = correlation

    return state
