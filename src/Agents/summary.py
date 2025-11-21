from src.llm.model import model
from src.state import MLState
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def summary(state: MLState):
    """
    Generates a textual summary/report of the dataset and model results using an LLM.
    """

    df = state["df"]
    shape = df.shape
    target = state["target"]
    problem_type = state["problem_type"]
    results = state["results"]
    best_model_name = state["best_model_name"]
    metrics = state["metrics"]

    template = PromptTemplate(
        template="""
You are an expert data scientist. Based on the following information, generate a structured AutoML summary report.

### Dataset Overview
- Shape: {shape}
- Target: {target}
- Problem Type: {problem_type}

### Model Comparison
{results}

### Best Model
- Name: {best_model_name}
- Metrics: {metrics}

### Write a clear professional report covering:
1. Dataset characteristics  
2. Target type and prediction problem  
3. Explanation of model performance  
4. Why the best model performed best  
5. Actionable recommendations to improve the model  
""",
        input_variables=[
            "shape",
            "target",
            "problem_type",
            "results",
            "best_model_name",
            "metrics"
        ]
    )

    parser = StrOutputParser()

    chain = template | model | parser

    output = chain.invoke({
        "shape": shape,
        "target": target,
        "problem_type": problem_type,
        "results": results,
        "best_model_name": best_model_name,
        "metrics": metrics
    })

    return {"summary": output}



