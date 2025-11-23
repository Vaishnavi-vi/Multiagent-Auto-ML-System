from src.llm.model import model
from src.state import MLState
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def summary(state: MLState):
    df = state["df"]
    shape = df.shape
    target = state["target"]
    problem_type = state["problem_type"]
    numeric_features = state.get("numeric_features",[])
    categorical_features = state.get("categorical_features",[])
    missing_values = state.get("missing_values",[])
    basic_stats = state.get("basic_stats",[])
    correlation=state.get("correlation",[])
    results = state.get("results",[])
    best_model_name = state["best_model_name"]
    metrics = state["metrics"]

    # Convert lists/dicts to string for LLM
    numeric_features_str = ", ".join(numeric_features)
    categorical_features_str = ", ".join(categorical_features)
    
    template = PromptTemplate(
        template="""
You are an expert data scientist. Based on the following information, generate a structured AutoML summary report.

### Dataset Overview
- Shape: {shape}
- Target: {target}
- Problem Type: {problem_type}
- Numeric features: {numeric_features} 
- Categorical features: {categorical_features}
- Missing values: {missing_values}
- Basic stats: {basic_stats}
- correlation:{correlation}

### Feature Engineering
"Applied StandardScaler and SimpleImputer to numerical columns and OneHotEncoder to categorical columns but not for target columns"

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

Prepared by Vaishnavi Barolia
""",
        input_variables=[
            "shape",
            "target",
            "problem_type",
            "numeric_features",
            "categorical_features",
            "missing_values",
            "basic_stats",
            "results",
            "best_model_name",
            "metrics","correlation"
        ]
    )

    parser = StrOutputParser()
    chain = template | model | parser

    output = chain.invoke({
        "shape": shape,
        "target": target,
        "problem_type": problem_type,
        "numeric_features": numeric_features_str,
        "categorical_features": categorical_features_str,
        "missing_values": missing_values,
        "basic_stats": basic_stats,
        "correlation":correlation,
        "results": results,
        "best_model_name": best_model_name,
        "metrics": metrics
    })

    return {"summary": output}




