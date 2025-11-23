from src.state import MLState
import pandas as pd

def regression_cleaning_agent(state:MLState):
    df=state["df"]
    target=state["target"]
    
    numeric_features = df.select_dtypes(include="number").columns.to_list()
    categorical_features = df.select_dtypes(exclude="number").columns.to_list()
    
    text_cols = df.select_dtypes(include=['object']).columns
    text_cols = [col for col in text_cols if col != target]

    df_numeric_only = df.drop(columns=text_cols)

    if df_numeric_only.shape[1] > 1:  
        df = df_numeric_only   

    
    missing_values = df.isnull().sum().to_dict()
    basic_stats = df.describe().to_dict()
    correlation = df.corr().to_dict()
    
    #Normalize column name
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    
    #Drop fully null columns
    df = df.dropna(axis=1, how="all")
    
    #Drop rows with target is equal to null
    df = df[df[target].notna()].reset_index(drop=True)
    
    #Drop duplicates
    df=df.drop_duplicates(keep="first").reset_index(drop=True)
    
    #Drop Constant Columns
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)
    
    #Drop when null is greater than 0.8
    df = df.loc[:, df.isnull().mean() < 0.8]
    
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(str)
    
    #outliers
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in num_cols:
        skew = df[col].skew()

        if abs(skew) < 1:
            # Light skew → 5%–95% clipping
            low_q, high_q = 0.05, 0.95
        else:
            # Heavy skew → 1%–99% clipping
            low_q, high_q = 0.01, 0.99

        # compute limits
        lower = df[col].quantile(low_q)
        upper = df[col].quantile(high_q)

        # clip the values
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

