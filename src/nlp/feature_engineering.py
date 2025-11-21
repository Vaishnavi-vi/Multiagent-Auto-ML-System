from src.state import MLState


def nlp_feature_engineering_agent(state: MLState):
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = state["df"]

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["text_clean"])

    state["features"] = X
    state["vectorizer"] = vectorizer

    return {"features":X,"vectorizer":vectorizer}
