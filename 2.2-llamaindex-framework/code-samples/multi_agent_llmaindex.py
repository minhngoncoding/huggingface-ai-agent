import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from llama_index.core.agent.workflow import ReActAgent, AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

load_dotenv(find_dotenv())

hf_token = os.getenv("HF_TOKEN")


def add(a: int, b: int) -> int:
    """Add two integer numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """multiply two integer numbers."""
    return a * b


llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-7B-Instruct")

multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two numbers.",
    system_promt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

adding_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two numbers.",
    system_promt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

workflow = AgentWorkflow(
    agents=[multiply_agent, adding_agent], root_agent="multiply_agent"
)


async def main():
    # Calling the workflow
    response = await workflow.run(user_msg="Can you add 5 and 3?")
    print(f"Final Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())
