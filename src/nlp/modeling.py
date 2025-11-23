from src.state import MLState
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def nlp_modeling_agent(state: MLState):
    le=LabelEncoder()
    y_train=le.fit_transform(state["y_train"])
    y_test=le.transform(state["y_test"])
    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "naive_bayes": MultinomialNB(),
        "svm":LinearSVC()
    }
    results = {}

    for name, model in models.items():
        model.fit(state["x_train"],y_train)
        y_pred = model.predict(state["x_test"])
        acc = accuracy_score(y_test, y_pred)
        pre=precision_score(y_test,y_pred,average="weighted")
        recall=recall_score(y_test,y_pred,average="weighted")
        f1=f1_score(y_test,y_pred,average="weighted")

        # store result for comparison
        results[name] = {
            "accuracy_score": acc,
            "precision_score":pre,
            "recall_score":recall,
            "f1_score":f1
        }
        
        if "best_model" not in state or acc > state.get("best_score", 0):
            state["best_model_name"] = name
            state["best_score"] = acc
            state["best_model"]=model

    state["results"] = results
    state["metrics"] = {name: res["accuracy_score"] for name, res in results.items()}
    state["y_train"]=y_train
    state["y_test"]=y_test

    return state

        
        

        





    
    
    
    

