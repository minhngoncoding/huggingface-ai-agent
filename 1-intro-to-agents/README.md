# Unit 1 - Introduction to Agents

Status: In progress
Pinned: No
Created time: March 17, 2026 1:22 PM
Projects: HuggingFace AI Agent (https://www.notion.so/HuggingFace-AI-Agent-326dfdb107c680859a4bcfd7b43f71dc?pvs=21)
Archive: No
Settings: comnamcangu (https://www.notion.so/comnamcangu-2eddfdb107c6811d9848cffa963de89c?pvs=21)
Insights: Insights (https://www.notion.so/Insights-2eddfdb107c6818fb904dc611a7e9805?pvs=21)
Total EXP: 0
XP: 💎 +0 XP

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image.png)

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%201.png)

# 1. What is an Agent?

- **Core Definition:** An AI Agent is a system that uses an AI model to interact with its environment in order to achieve a user-defined objective. It actively combines reasoning, planning, and the execution of actions (often using external tools) to fulfill tasks.
- **The Structure of an Agent:** An agent can be conceptualized as having two main parts:
    - **The Brain (AI Model):** This is typically a Large Language Model (LLM) like GPT-4, Llama, or Gemini. The brain handles the thinking, reasoning, and planning, and decides which actions to take based on the current situation.
    - **The Body (Capabilities and Tools):** This represents everything the agent is equipped to do. An agent's scope of actions depends entirely on the tools it has been provided with.
- **How Agents Take Action:** On their own, LLMs can only process and generate text. To allow the AI to interact with the outside world (like generating images or sending emails), developers provide them with **Tools**. The LLM generates text or code to trigger these tools, successfully fulfilling the desired task.
- **The Spectrum of Agency:** Agents operate on a continuous spectrum of autonomy. This ranges from simple routing decisions to complex multi-step loops where the agent continuously executes steps until an objective is met, or even multi-agent setups where one agent triggers another.
- **Real-World Examples:**
    - **Personal Virtual Assistants:** Such as Siri or Alexa, which analyze context and use tools to set reminders, send messages, or interact with digital environments on behalf of the user.
    - **Customer Service Chatbots:** Agents that assist users by troubleshooting, answering questions, or completing transactions in internal databases.
    - **Video Game NPCs:** Non-playable characters that adapt dynamically to player interactions and generate nuanced dialogue instead of relying on fixed, rigid scripts.

**In summary**, the core of an AI Agent lies in three capabilities: **understanding natural language**, **reasoning and planning**, and **interacting with its environment** (gathering information, acting, and observing the results).

# 2. What are LLMs?

- **Core Definition:** **Large Language Models (LLMs)** are AI models with millions or billions of parameters that excel at understanding and generating human language. In the context of AI Agents, the LLM acts as the "brain," interpreting instructions, maintaining conversation context, planning, and deciding which tools to use.
- **The Transformer Architecture:** Most modern LLMs are built on the Transformer architecture, which relies on the "Attention" algorithm. There are three main types of transformers:
    - **Encoders (e.g., BERT):** Used for tasks like text classification and semantic search.
    - **Decoders (e.g., Llama):** Focus on generating text one token at a time. **Most prominent LLMs are decoder-based**.
    - **Seq2Seq / Encoder-Decoder (e.g., T5):** Combines both, typically used for translation or summarization.
- **Tokens and Special Tokens:** LLMs process text in sub-word units called **tokens** rather than whole words. Models also rely on **special tokens** (which vary across different model providers) to structure prompts and mark the beginning or end of sequences, such as the End of Sequence (EOS) token.
- **Next Token Prediction and Decoding:** The primary objective of an LLM is simply to predict the next token. They are **autoregressive**, meaning the output of one pass becomes the input for the next, continuously looping until the model generates an EOS token. To select the next token, models use decoding strategies ranging from simple greedy selection (picking the max score) to more advanced methods like **beam search**, which explores multiple candidate sequences simultaneously to find the best overall score.
- **Attention and Context Length:** When predicting the next word, the attention mechanism helps the model determine which previous words in the prompt carry the most meaning. The maximum number of tokens an LLM can analyze at once is called its context length.
- **Training Process:** LLMs are first **pre-trained** on massive text datasets using unsupervised learning to learn language structures and patterns. After that, they are **fine-tuned** using supervised learning to perform specific tasks, such as following conversational structures or using tools.

# 3. Messages and Special Tokens

- **The Illusion of Memory:** While chat interfaces make it seem like models remember the conversation, they actually do not. Every time you interact, all previous messages are concatenated into a single, long prompt before being fed to the model.
- **System Messages (System Prompts):** These are persistent instructions that define how the model should behave. For AI Agents, the system message is extremely important because it provides the model with information about available tools, instructions on how to format actions, and guidelines on how to segment its thought process.

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%202.png)

- **User and Assistant Conversations:** These are the alternating messages between the human and the AI. Keeping this history helps maintain context for multi-turn conversations.
- **Special Tokens:** These are specific markers used by models to delimit where a user or assistant turn starts and ends, such as the End Of Sequence (EOS) token. Because different models (like Llama 3, GPT-4, or SmolLM2) use different special tokens, the formats can vary significantly between providers.
- **Chat Templates:** Chat templates act as the bridge between your conversational messages and the specific formatting requirements of your chosen LLM. They automatically structure the communication and apply the correct special tokens so the model can understand it.
- **Base vs. Instruct Models:** Base models are only trained on raw text to predict the next token. Instruct models, however, are specifically fine-tuned to follow instructions and engage in conversations, which requires prompts to be formatted consistently using chat templates.

## 3.1 Chat Templates

Chat templates act as the **bridge between conversational messages and the specific formatting requirements of your chosen Large Language Model (LLM)**.

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%203.png)

Here is a more specific breakdown of how they work and why they are necessary:

- **Solving the "Illusion of Memory":** While chat interfaces make it look like the AI remembers your back-and-forth conversation, the model actually does not; it reads the entire conversation history from scratch every time. A chat template is responsible for taking your list of separate messages (System, User, and Assistant roles) and **concatenating them into a single, continuous string** (a prompt) that the model can process.
- **Managing Unique Special Tokens:** Every LLM has its own unique set of "special tokens" used to mark where a message starts and ends. For example, the SmolLM2 model uses tokens like `<|im_start|>` and `<|im_end|>`, whereas the Llama 3.2 model uses `<|start_header_id|>` and `<|eot_id|>`. The chat template ensures that the correct delimiters are automatically applied for the specific model you are using.
- **Crucial for Instruct Models:** While base models are only trained on raw text to predict the next word, **Instruct Models are fine-tuned specifically to engage in conversations**. To trigger this conversational behavior correctly, you must format your prompts using the exact chat template the model was trained on. A common standard format used for structuring these roles is *ChatML*.
- **Under the Hood (Jinja2):** In the `transformers` library, chat templates are implemented using **Jinja2 code**. This code acts as a blueprint that dictates exactly how to transform a list of JSON messages into the final textual representation that the model expects.
- **Practical Application:** In code, the easiest way to format your conversation properly is by loading the model's tokenizer and calling the **`apply_chat_template`** function. This function takes your list of messages, applies the Jinja2 template, adds all the necessary special tokens, and returns a rendered prompt that is ready to be fed directly into the LLM.

# 4. What are Tools?

- **What are AI Tools?:** A tool is a function given to a Large Language Model (LLM) that fulfills a clear objective. Tools are essential because they give the LLM extra capabilities beyond its static training data, allowing it to perform tasks like math calculations, execute code, or fetch up-to-date information via web searches and APIs.
    
    ![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%204.png)
    
- **Core Components of a Tool:** To be effective, a tool must contain a clear textual description of what it does, a callable function (the actual code that performs the action), specific arguments with their data types, and optionally, defined outputs.
- **How Tools Work:** LLMs cannot actually execute code or call APIs themselves; they only generate text. When you "give" a tool to an LLM, you are actually just describing the tool and its required inputs in the **system prompt**. When the LLM decides the tool is needed, it generates a text-based command (e.g., `call weather_tool('Paris')`). The **Agent** then reads this text, executes the tool on the LLM's behalf, and feeds the resulting data back to the LLM as a new message.

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

- **Auto-formatting Tools:** Manually writing out the strict formats (like JSON) needed to describe tools to an LLM can be fragile and tedious. To solve this, developers can use Python decorators (such as `@tool`) that automatically extract the tool's name, docstring description, and argument type hints directly from the source code to build the system prompt automatically.
- **Model Context Protocol (MCP):** This is an open protocol that standardizes how applications provide tools to LLMs. It allows any framework implementing MCP to leverage a growing list of pre-built tools, making it easy to switch between LLM providers and frameworks without rewriting the tool interfaces.

# 5. **Thought-Action-Observation cycle**

The **Thought-Action-Observation cycle** is the core workflow that defines how AI agents operate and solve complex tasks. You can think of this cycle as a `while loop` in programming: the agent continuously loops through these three steps until its user-defined objective is fully completed.

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%205.png)

In many Agent frameworks, **the rules and guidelines are embedded directly into the system prompt**, ensuring that every cycle adheres to a defined logic.

In a simplified version, our system prompt may look like this:

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%206.png)

Here is a breakdown of the three core components:

- **Thought (Internal Reasoning):** This is the planning phase where the LLM acts as the agent's "brain." It analyzes the user's request, breaks the problem down into manageable steps, and decides what the next move should be.
- **Action (Tool Usage):** Based on its reasoning, the agent acts by calling a specific tool (such as an API, web search, or calculator) and passing the necessary arguments to it.
- **Observation (Feedback from the Environment):** After the tool executes, the agent receives the results (or error messages). This feedback is appended to the agent's memory as an observation, which the agent reflects on to determine its next move.

**How it works in practice:**
The course illustrates this using an example of "Alfred," a weather agent. If a user asks for the current weather in New York, Alfred first **thinks** that he needs to fetch up-to-date data using his available weather tool. He then takes **action** by generating a JSON command to call the `get_weather` tool with the location parameter set to "New York". The API returns an **observation** (e.g., "partly cloudy, 15°C"), which Alfred reads. Reflecting on this new data (an **updated thought**), Alfred realizes he has enough information to fulfill the objective and executes a **final action** to output the answer to the user.

**Why this cycle is essential:**
The guidelines for this cycle are usually embedded directly into the agent's system prompt. This continuous loop, often referred to as the **ReAct (Reasoning + Acting) cycle**, gives agents three major advantages:

1. **Iterative Problem Solving:** If an observation indicates an error or missing information, the agent does not just fail; it can re-enter the cycle, correct its approach, and try a different strategy.
2. **Tool Integration:** It allows the agent to bypass the static limitations of its training data and interact with the real world to retrieve live data.
3. **Dynamic Adaptation:** Each cycle allows the agent to incorporate fresh, real-world feedback into its reasoning, ensuring its final answer is highly accurate and contextually aware.

# 6. **Thought: Internal Reasoning and the ReAct Approach**.

- **What are Thoughts?:** Thoughts represent the agent's internal reasoning and planning processes. By acting as an "inner monologue" within the prompt, thoughts help the agent analyze information, break down complex problems into manageable steps, reflect on past experiences, and adjust its plans based on new observations.

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%207.png)

- **Types of Thoughts:** An agent's internal reasoning can take many forms, including planning, analysis, decision making, problem-solving, self-reflection, and prioritization.
- **Chain-of-Thought (CoT) Prompting:** This is a prompting technique that guides a model to think through a problem step-by-step before producing a final answer, usually initiated by the phrase *"Let's think step by step"*. CoT is highly effective for logical or mathematical tasks, but it relies purely on internal reasoning **without interacting with external tools**.

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%208.png)

- **The ReAct Approach (Reasoning + Acting):** ReAct takes CoT a step further by combining step-by-step reasoning with real-world actions. It prompts the model to interleave its thoughts with actions (tool usage) and observations. While CoT is best for internal logic, ReAct is essential for information-seeking and dynamic, multi-step tasks that require external data.

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%209.png)

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%2010.png)

# 7. **Actions: Enabling the Agent to Engage with Its Environment**.

- **What are Actions?:** Actions are the concrete steps an AI agent takes to interact with its environment, bridging its internal reasoning with real-world execution. Actions can serve many purposes, such as gathering information (web searches), using tools (API calls, running calculations), interacting with environments (manipulating digital interfaces), or communicating with users and other agents.
- **Types of Agents:** Agents can be categorized by how they output their actions:
    - **JSON Agents:** The action and its arguments are specified in a JSON format.
    - **Function-calling Agents:** A subcategory of JSON agents where the underlying LLM has been explicitly fine-tuned to generate a new message for each action.
    - **Code Agents:** Instead of outputting a simple JSON object, the agent generates an executable block of code, typically in Python.
- **The "Stop and Parse" Approach:** Because an LLM can only generate text, there needs to be a mechanism to turn that text into a real action. This is achieved through the stop and parse approach:
    1. **Generation:** The agent outputs its intended action in a clear, predetermined format (like JSON or code).
    2. **Halting:** Once the action is fully defined, the LLM *must stop* generating new tokens to prevent erroneous output and hand control back to the agent framework.
    3. **Parsing:** An external parser reads the formatted text, determines which tool to call, and extracts the required parameters to execute it.

```python
Thought: I need to check the current weather for New York.
Action :
{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}
```

- **Code Agents:** Using code to define actions offers several benefits over JSON, including greater expressiveness (the ability to use loops, conditionals, and nested logic), modularity, easier debugging, and direct integration with external libraries. However, executing LLM-generated code poses security risks, which is why it is recommended to use frameworks with built-in safeguards, such as `smolagents`.

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%2011.png)

```python
# Code Agent Example: Retrieve Weather Information
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

# Execute the function and prepare the final answer
result = get_weather("New York")
final_answer = f"The current weather in New York is: {result}"
print(final_answer)
```

# 8. **Observe: Integrating Feedback to Reflect and Adapt**

- **What are Observations?:** Observations are how an AI agent perceives the consequences of the actions it just took. They act as signals from the environment—such as data from an API, error messages, or system logs—that fuel and guide the agent's next thought process.
- **The Observation Phase:** During this phase, the agent performs three key steps:
    1. **Collects Feedback:** It receives data confirming whether its intended action succeeded or failed.
    2. **Appends Results:** It integrates this new real-world information into its existing context, effectively updating its memory.
    3. **Adapts its Strategy:** It uses this updated context to refine its subsequent thoughts and decide on the next best action.
- **Types of Observations:**

![image.png](Unit%201%20-%20Introduction%20to%20Agents/image%2012.png)

- **How the Framework Handles It:** The mechanics behind this are straightforward. After the agent decides on an action, the framework parses the action to identify the function and arguments, executes that function, and then appends the exact result directly into the prompt as an "Observation". This iterative feedback loop ensures the agent constantly learns and adjusts based on real-world outcomes to reach its goal.

With the Thought-Action-Observation cycle fully covered, the theoretical part of Unit 1 is complete! The remaining sections of the unit focus on putting these concepts into practice by exploring a **Dummy Agent Library** and **Creating Your First Agent Using smolagents**