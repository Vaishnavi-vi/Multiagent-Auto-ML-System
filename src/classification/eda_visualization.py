from src.state import MLState

import matplotlib.pyplot as plt
import seaborn as sns

def classification_eda_visualization_agent(state: MLState):
    df = state["df"]
    
    # Numerical features
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    # Plot histograms for numerical columns
    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.show()
    
    # Plot bar plots for categorical columns
    for col in cat_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(x=col, data=df)
        plt.title(f"Counts of {col}")
        plt.xticks(rotation=45)
        plt.show()
    
    
    return state
