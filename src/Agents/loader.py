from src.state import MLState
import pandas as pd


def loader_agent(state:MLState):
    path=state.get("csv_path")
    dff = pd.read_csv(path)
    dataframe = pd.DataFrame(dff)
  
    return {"df":dataframe}