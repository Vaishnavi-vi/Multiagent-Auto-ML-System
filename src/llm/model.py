from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm(api_key: str = None):
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-20b",
        temperature=0.7,
        task="text_generation",
        model_kwargs={"api_key": api_key}
    )
    return ChatHuggingFace(llm=llm_endpoint)

model = get_llm(api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"))