from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
from src.state import MLState
from src.workflow.pipeline import workflow


from fastapi import FastAPI

app = FastAPI(title="AutoML + NLP API", version="1.0")


@app.get("/health")
async def health_check():
    """
    Simple health check to verify the API is running
    """
    return {
        "status": "ok",
        "message": "API is up and running "
    }




@app.get("/about")
async def about():
    """
    Returns information about this API
    """
    return {
        "name": "Multi-Agent AutoML + NLP API",
        "version": "1.0",
        "description": (
            "This API allows users to upload a CSV, detect problem type (regression, "
            "classification, NLP), run automated ML pipelines, and return results including "
            "EDA, model comparison, metrics, and summary."
        ),
        "author": "Your Name or Organization"
    }

@app.post("/run_automl")
async def run_automl(
    file: UploadFile = File(...),
    target: str = Form(...)
):
    """
    Run the AutoML workflow with an uploaded CSV
    """
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

        df = pd.read_csv(file.file)

        if target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target}' not found in CSV. Available columns: {list(df.columns)}"
            )

        initial_state: MLState = {
            "csv_path": None,  # optional, we have df in memory
            "target": target,
            "df": df,
            "problem_type": None,
            "cleaned": False,
            "x": None,
            "y": None,
            "features": None,
            "vectorizer": None,
            "X_train": None,
            "X_test": None,
            "y_train": None,
            "y_test": None,
            "y_pred": None,
            "preprocess": None,
            "results": {},
            "best_model_name": None,
            "best_model": None,
            "metrics": None,
            "visualization_done": False,
            "summary": ""
        }

        output = workflow.invoke(initial_state)

        return {
            "problem_type": output.get("problem_type"),
            "metrics": output.get("metrics"),
            "best_model_name": output.get("best_model_name"),
            "results": output.get("results"),
            "summary": output.get("summary")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

