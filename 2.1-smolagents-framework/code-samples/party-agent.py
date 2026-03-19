from dotenv import load_dotenv, find_dotenv
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, tool

load_dotenv(find_dotenv())

# Model Configuration
model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5-coder:14b",
    api_base="http://localhost:11434",
    num_ctx=8192,
)


# Tool to suggest a menu based on the occasion
@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion (str): The type of occasion for the party. Allowed values are:
                        - "casual": Menu for casual party.
                        - "formal": Menu for formal party.
                        - "superhero": Menu for superhero party.
                        - "custom": Custom menu.
    """
    if occasion == "casual":
        return "Pizza, snacks, and drinks."
    elif occasion == "formal":
        return "3-course dinner with wine and dessert."
    elif occasion == "superhero":
        return "Buffet with high-energy and healthy food."
    else:
        return "Custom menu for the butler."


# Agent Configuration
agent = CodeAgent(
    tools=[DuckDuckGoSearchTool(), suggest_menu],
    model=model,
    max_steps=5,
    verbosity_level=1,
    additional_authorized_imports=["datetime"],
)

# agent.run("Prepare a formal menu for the party")

# agent.push_to_hub("minhngon1520/PartyAgent")
agent.run(
    """
    Alfred needs to prepare for the party. Here are the tasks:
    1. Prepare the drinks - 30 minutes
    2. Decorate the mansion - 60 minutes
    3. Set up the menu - 45 minutes
    4. Prepare the music and playlist - 45 minutes

    If we start right now, at what time will the party be ready?
    """
)
