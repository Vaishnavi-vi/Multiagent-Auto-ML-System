from src.state import MLState

def nlp_eda_agent(state: MLState):
    import matplotlib.pyplot as plt

    df = state["df"]
    target = state["target"]
    text_col = target  # for NLP dataset target column is text data

    plt.figure(figsize=(6, 6))
    df[target].value_counts().plot(kind="pie", autopct="%1.1f%%")
    plt.ylabel("")
    plt.title("Target Distribution")
    plt.close()


    df["text_length"] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    plt.figure(figsize=(8, 4))
    df["text_length"].plot(kind="hist", bins=50)
    plt.title("Sentence Length Distribution")
    plt.xlabel("Number of Words")
    plt.close()
    
    
    return state



   
