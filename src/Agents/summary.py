from src.llm.model import model
from src.state import MLState
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

def summary(state:MLState):
    """
    Generates a textual summary/report of the dataset and target using an LLM.
    """
    df=state["df"]
    shape=df.shape
    target=state["target"]
    problem_type=state["problem_type"]
    results=state["results"]
    best_model_name=state["best_model_name"]
    best_model=state["best_model"]
    metrics=state["metrics"]

    template = PromptTemplate(
        template="""You are an expert data scientist.Dataset shape: {shape},Target column: {target},problem_type:{problem_type},Model results: {results} with best_model_name as {best_model_name} best_model:{best_model} and metrics:{metrics}
        Write a concise report explaining:
        - Dataset characteristics
        - Target type and prediction problem
        - Model performance metrics
        - Recommendations for improvement
""",
        input_variables=["df.shape", "target", "results","problem_type","best_model","best_mdodel_name","metrics"]
    )
    
    parser=StrOutputParser()
    
    chain=template|model|parser
    
    output=chain.invoke({"shape":shape,"target":target,"results":results,"problem_type":problem_type,"best_model":best_model,"best_model_name":best_model_name,"metrics":metrics})
    
    return {"summary":output}