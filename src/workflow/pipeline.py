from langgraph.graph import StateGraph,START,END
from src.state import MLState
from src.Agents.loader import loader_agent
from src.Agents.problem_detector import problem_detector_agent
from src.Agents.summary import summary
from typing import Literal

from src.classification.cleaning_agent import classification_cleaning_agent
from src.classification.eda_visualization import classification_eda_visualization_agent
from src.classification.evaluation import classification_evaluation_agent
from src.classification.feature_engineering import classification_feature_engineering_agent
from src.classification.modeling import classification__modeling_agent
from src.classification.train_test_split import classification_train_test_split_agent
from src.classification.visual_evaluation import classification_model_visualization_agent

from src.nlp.cleaning_agent import nlp_cleaning_agent
from src.nlp.eda_visualization import nlp_eda_agent
from src.nlp.evaluation import nlp_evaluation_agent
from src.nlp.feature_engineering import nlp_feature_engineering_agent
from src.nlp.modeling import nlp_modeling_agent
from src.nlp.train_test_split import nlp_train_test_split
from src.nlp.visual_evaluation import nlp_model_visualization_agent

from src.Regression.cleaning_agent import regression_cleaning_agent
from src.Regression.eda_visualization import regression_eda_visualization_agent
from src.Regression.evaluation import regression_evaluation_agent
from src.Regression.feature_engineering import regression_feature_engineering_agent
from src.Regression.modelling import regression_modeling_agent
from src.Regression.train_test_split import regression_train_test_split_agent
from src.Regression.visual_evaluation import regression_model_visualization_agent


def check_condition(state:MLState)->Literal["regression_cleaning_agent","classification_cleaning_agent","nlp_cleaning_agent"]:
    if state["problem_type"]=="classification":
        return "classification_cleaning_agent"
    elif state["problem_type"]=="nlp":
        return "nlp_cleaning_agent"
    else:
        return "regression_cleaning_agent"


graph = StateGraph(MLState)

# Add nodes
graph.add_node("loader_agent",loader_agent)
graph.add_node("problem_detector_agent",problem_detector_agent)
graph.add_node("regression_cleaning_agent",regression_cleaning_agent)
graph.add_node("regression_eda_visualization_agent",regression_eda_visualization_agent)
graph.add_node("regression_feature_engineering_agent",regression_feature_engineering_agent)
graph.add_node("regression_train_test_split_agent",regression_train_test_split_agent)
graph.add_node("regression_modeling_agent",regression_modeling_agent)
graph.add_node("regression_evaluation_agent",regression_evaluation_agent)
graph.add_node("regression_model_visualization_agent",regression_model_visualization_agent)

graph.add_node("nlp_cleaning_agent",nlp_cleaning_agent)
graph.add_node("nlp_eda_agent",nlp_eda_agent)
graph.add_node("nlp_feature_engineering_agent",nlp_feature_engineering_agent)
graph.add_node("nlp_train_test_split",nlp_train_test_split)
graph.add_node("nlp_modeling_agent",nlp_modeling_agent)
graph.add_node("nlp_evaluation_agent",nlp_evaluation_agent)
graph.add_node("nlp_model_visualization_agent",nlp_model_visualization_agent)
graph.add_node("summary",summary)

graph.add_node("classification_cleaning_agent",classification_cleaning_agent)
graph.add_node("classification_eda_visualization_agent",classification_eda_visualization_agent)
graph.add_node("classification_feature_engineering_agent",classification_feature_engineering_agent)
graph.add_node("classification_train_test_split_agent",classification_train_test_split_agent)
graph.add_node("classification_modeling_agent",classification__modeling_agent)
graph.add_node("classification_evaluation_agent",classification_evaluation_agent)
graph.add_node("classification_model_visualization_agent",classification_model_visualization_agent)


graph.add_edge(START,"loader_agent")
graph.add_edge("loader_agent","problem_detector_agent")
graph.add_conditional_edges("problem_detector_agent",check_condition)
graph.add_edge("regression_cleaning_agent","regression_eda_visualization_agent")
graph.add_edge("regression_eda_visualization_agent","regression_feature_engineering_agent")
graph.add_edge("regression_feature_engineering_agent","regression_train_test_split_agent")
graph.add_edge("regression_train_test_split_agent","regression_modeling_agent")
graph.add_edge("regression_modeling_agent","regression_evaluation_agent")
graph.add_edge("regression_evaluation_agent","regression_model_visualization_agent")
graph.add_edge("regression_model_visualization_agent","summary")
graph.add_edge("summary",END)

graph.add_edge("classification_cleaning_agent","classification_eda_visualization_agent")
graph.add_edge("classification_eda_visualization_agent","classification_feature_engineering_agent")
graph.add_edge("classification_feature_engineering_agent","classification_train_test_split_agent")
graph.add_edge("classification_train_test_split_agent","classification_modeling_agent")
graph.add_edge("classification_modeling_agent","classification_evaluation_agent")
graph.add_edge("classification_evaluation_agent","classification_model_visualization_agent")
graph.add_edge("classification_model_visualization_agent","summary")
graph.add_edge('summary',END)


graph.add_edge("nlp_cleaning_agent","nlp_eda_agent")
graph.add_edge("nlp_eda_agent","nlp_feature_engineering_agent")
graph.add_edge("nlp_feature_engineering_agent","nlp_train_test_split")
graph.add_edge("nlp_train_test_split","nlp_modeling_agent")
graph.add_edge("nlp_modeling_agent","nlp_evaluation_agent")
graph.add_edge("nlp_evaluation_agent","nlp_model_visualization_agent")
graph.add_edge("nlp_model_visualization_agent","summary")
graph.add_edge('summary',END)

workflow=graph.compile()