from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import pandas as pd
from src.state import MLState
from src.workflow.pipeline import workflow
import io,base64
from io import BytesIO
import matplotlib.pyplot as plt

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
async def run_automl(file: UploadFile = File(...), target: str = Form(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

        # Read CSV in-memory
        df = pd.read_csv(file.file)

        if target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target}' not found. Columns: {list(df.columns)}"
            )

        # Initialize state
        initial_state: MLState = {
            "csv_path": None,
            "df": df,
            "target": target,
            "problem_type": None,
            "numeric_features": [],
            "categorical_features": [],
            "preprocess": None,
            "x": None,
            "y": None,
            "x_train": None,
            "x_test": None,
            "y_train": None,
            "y_test": None,
            "y_pred": None,
            "results": {},
            "best_model_name": None,
            "best_model": None,
            "metrics": {},
            "plots": {},
            "summary": ""
        }

        # Run your workflow
        output = workflow.invoke(initial_state)
        safe_plots=output.get("plots",{})
     
        return {
            "problem_type": output.get("problem_type"),
            "metrics": output.get("metrics"),
            "best_model_name": output.get("best_model_name"),
            "results": output.get("results"),
            "plots":safe_plots,
            "summary": output.get("summary")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))