from src.state import MLState

def classification_cleaning_agent(state:MLState):
    df=state["df"]
    target=state["target"]
    
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
    
    return {"df":df,"cleaned":True}