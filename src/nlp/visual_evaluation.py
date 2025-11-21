from src.state import MLState
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
def nlp_model_visualization_agent(state: MLState):
    
    y_true = state.get("y_test")
    y_pred = state.get("y_pred")  
    
    y_true = state["y_test"]
    y_pred = state["y_pred"]
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
    
    return state