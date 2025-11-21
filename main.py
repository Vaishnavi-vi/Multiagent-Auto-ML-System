from src.Agents.loader import loader_agent
from src.Agents.problem_detector import problem_detector_agent
from src.workflow.pipeline import workflow

if __name__ == "__main__":

    # ---- Initial State ----
    initial_state = ({
        "csv_path": "D:\\Desktop\\Multiagent-Auto-ML-System\\Experiments\\fake_and_real_news.csv",      # path to your dataset
        "target": "label",           # target column name
    })


    # ---- Run the workflow ----
    result = workflow.invoke(initial_state)

    # ---- Final Output ----
    print("\n=============================")
    print("          FINAL REPORT       ")
    print("=============================\n")

    if "summary" in result:
        print(result["summary"])
    else:
        print("No summary generated.")