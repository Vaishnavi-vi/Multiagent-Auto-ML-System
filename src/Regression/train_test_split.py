from sklearn.model_selection import train_test_split
from src.state import MLState

def regression_train_test_split_agent(state: MLState):
    x, y = state["x"], state["y"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42)
    
    state["x_train"]=x_train
    state["x_test"]=x_test
    state["y_train"]=y_train
    state["y_test"]=y_test
 
    
    return state
    
    

