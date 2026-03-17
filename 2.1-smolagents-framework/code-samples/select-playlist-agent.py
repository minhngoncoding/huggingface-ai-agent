import os
from dotenv import load_dotenv, find_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel

load_dotenv(find_dotenv())

# 1. Model Configuration 
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    num_ctx=8192
)

# 2. Agent Configuration 
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool()], 
    model=model,
    max_steps=5,        
    verbosity_level=1  
)

agent.run(
    "Search for 5 popular Vietnamese 'sad' songs for someone with a broken heart.")
