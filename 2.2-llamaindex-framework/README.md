# Unit 2.2- The LLamaIndex framework

Status: Inbox
Pinned: No
Created time: March 19, 2026 2:19 PM
Projects: HuggingFace AI Agent  (https://www.notion.so/HuggingFace-AI-Agent-326dfdb107c680859a4bcfd7b43f71dc?pvs=21)
Archive: No
Settings: comnamcangu (https://www.notion.so/comnamcangu-2eddfdb107c6811d9848cffa963de89c?pvs=21)
Insights: Insights (https://www.notion.so/Insights-2eddfdb107c6818fb904dc611a7e9805?pvs=21)
Total EXP: 0
XP: 💎 +0 XP

# Introduction to LlamaIndex

LlamaIndex is a comprehensive toolkit designed to help you create **LLM-powered agents over your data** by utilizing indexes and workflows.

### **Key Parts for Building Agents**

The framework focuses on four main pillars that act as the building blocks for agent creation:

- **Components:** These are the foundational building blocks, such as prompts, models, and databases. They often help connect LlamaIndex with other external libraries and tools.
- **Tools:** These are specialized components that provide specific capabilities, such as searching, calculating, or accessing external services, allowing the agent to perform real-world tasks.
- **Agents:** These are the autonomous decision-makers of the system. They coordinate the usage of various tools to accomplish complex goals.
- **Workflows:** These are step-by-step processes that link logic together. They provide a structured way to execute agentic behavior, sometimes without even needing explicit autonomous agents.

### **What Makes LlamaIndex Special?**

While LlamaIndex shares some functional similarities with other frameworks like `smolagents`, it offers several distinct advantages:

- **Clear Workflow System:** It helps break down agent decision-making step by step using a clear, **event-driven and async-first syntax**, making logic easy to compose and organize.
- **Advanced Document Parsing:** It features **LlamaParse**, a purpose-built (though paid) feature that seamlessly integrates advanced document parsing into the framework.
- **Many Ready-to-Use Components:** Because LlamaIndex has been established for a while, it boasts a large collection of thoroughly tested and reliable components, such as LLMs, retrievers, and indexes.
- **LlamaHub:** It provides a dedicated registry containing hundreds of tools, agents, and integrations that you can easily plug into your LlamaIndex projects.

# **What are components in LlamaIndex?**

Components are the building blocks that help an agent understand user requests and prepare, find, and use relevant information. A primary focus in LlamaIndex is the **`QueryEngine`** component, which acts as a Retrieval-Augmented Generation (RAG) tool. RAG helps agents overcome the limitations of an LLM's static training data by finding and retrieving relevant, up-to-date information from your specific data sources.

### **Creating a RAG pipeline using components**

Building a RAG pipeline involves five key stages:

- **Loading:** Ingesting data from files, databases, or APIs.
- **Indexing:** Creating searchable data structures, usually through vector embeddings.
- **Storing:** Saving the indexed data and metadata to avoid re-indexing.
- **Querying:** Using LLMs and data structures to search for answers.
- **Evaluation:** Checking the speed, accuracy, and faithfulness of the responses.

### **Loading and embedding documents**

Before data can be accessed, it must be loaded. This can be done using **`SimpleDirectoryReader`** for local files, **`LlamaParse`** for complex PDFs, or **LlamaHub** for various external sources. 

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()
```

Once loaded, an **`IngestionPipeline`** is used to break the documents down into smaller pieces called `Node` objects using a `SentenceSplitter`. These nodes are then transformed into numerical vector representations using an embedding model like `HuggingFaceEmbedding`.

```python
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ]
)

nodes = await pipeline.arun(documents=[Document.example()])
```

### **Storing and indexing documents**

To make the created nodes searchable, they must be stored in a vector database, such as **ChromaDB**. A **`VectorStoreIndex`** manages this process by embedding both the user's query and the data nodes into the same vector space, which allows the system to accurately find semantic matches.

```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0),
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)
```

```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
```

### **Querying a VectorStoreIndex with prompts and LLMs**

To ask questions, the index must be converted into a query interface. The main conversion options are:

- **`as_retriever`**: Returns a list of document chunks along with their similarity scores.
- **`as_query_engine`**: Processes single question-and-answer interactions and returns a written response.
- **`as_chat_engine`**: Maintains a conversation history for back-and-forth interactions.

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(
    llm=llm,
    response_mode="tree_summarize",
)
query_engine.query("What is the meaning of life?")
# The meaning of life is 42
```

### **Response Processing**

When querying, the engine uses a **`ResponseSynthesizer`** to formulate the LLM's final answer. It uses strategies such as **`refine`** (which makes separate LLM calls to refine the answer for each chunk), **`compact`** (the default strategy that concatenates chunks first to reduce LLM calls), or **`tree_summarize`** (which builds a tree structure to generate a detailed answer).

### **Evaluation and observability**

Because LLM outputs can be unpredictable, LlamaIndex includes built-in evaluators to measure response quality, such as the `FaithfulnessEvaluator`, `AnswerRelevancyEvaluator`, and `CorrectnessEvaluator`. Additionally, to understand how each component is performing and to gain deep system observability, you can integrate **LlamaTrace** (powered by Arize Phoenix).

- `FaithfulnessEvaluator`: Evaluates the faithfulness of the answer by checking if the answer is supported by the context.
- `AnswerRelevancyEvaluator`: Evaluate the relevance of the answer by checking if the answer is relevant to the question.
- `CorrectnessEvaluator`: Evaluate the correctness of the answer by checking if the answer is correct.

```python
from llama_index.core.evaluation import FaithfulnessEvaluator

query_engine = # from the previous section
llm = # from the previous section

# query index
evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response)
eval_result.passing
```

# **Using Tools in LlamaIndex**

Defining a clear set of tools is crucial for an agent's performance, as a clear interface makes it much easier for the LLM to understand and utilize them effectively. LlamaIndex features four main categories of tools: **`FunctionTool`**, **`QueryEngineTool`**, **`Toolspecs`**, and **`Utility Tools`**.

### **Creating a FunctionTool**

A **`FunctionTool`** provides a straightforward way to wrap any synchronous or asynchronous Python function and make it accessible to an agent. When creating one, providing a clear name and description is critical, because the LLM relies entirely on this metadata to figure out when and how to use the tool correctly.

```python
from llama_index.core.tools import FunctionTool

def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

tool = FunctionTool.from_defaults(
    get_weather,
    name="my_weather_tool",
    description="Useful for getting the weather for a given location.",
)
tool.call("New York")
```

### **Creating a QueryEngineTool**

A **`QueryEngineTool`** allows you to easily transform an existing `QueryEngine` (which we defined in the previous section) into a tool. Because agents are built on top of query engines, this functionality essentially allows agents to use other agents as tools.

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

db = chromadb.PersistentClient(path="./alfred_chroma_db")
chroma_collection = db.get_or_create_collection("alfred")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(llm=llm)
tool = QueryEngineTool.from_defaults(query_engine, name="some useful name", description="some useful description")
```

### **Creating Toolspecs**

**`Toolspecs`** are community-created collections of related tools that are designed to work harmoniously together for specific purposes—much like a well-organized mechanic's toolkit. For example, an accounting agent might use a toolspec that integrates spreadsheet, email, and calculation tools. You can install packages like the Google Toolspec or even leverage the **Model Context Protocol (MCP)** through an MCP Toolspec to plug into pre-built integrations.

```python
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()
```

### **Utility Tools**

Directly querying an API can sometimes return an excessive amount of data that might overflow the LLM's context window or run up token costs. **Utility Tools** help handle large amounts of data efficiently. The two main utility tools are:

- **`OnDemandToolLoader`**: This turns any existing LlamaIndex data loader into an agent tool. In a single call, it loads the data, indexes it (e.g., into a vector store), and then queries it on-demand based on a natural language string.
- **`LoadAndSearchToolSpec`**: This takes any existing tool as an input and splits it into two separate tools. First, an agent uses the "load" tool to call the underlying tool and index the output. Then, the agent uses the "search" tool to query that newly created index.

# **Using Agents in LlamaIndex**

![image.png](Unit%202%202-%20The%20LLamaIndex%20framework/image.png)

An Agent is a system that leverages an AI model to interact with its environment to achieve a user-defined objective by combining reasoning, planning, and action execution. LlamaIndex supports three main types of reasoning agents:

- **Function Calling Agents:** These work with AI models that are capable of calling specific functions.
- **ReAct Agents:** These can work with any AI that has chat or text completion endpoints and are highly effective at dealing with complex reasoning tasks.
- **Advanced Custom Agents:** These use more advanced methods to deal with complex tasks and customized workflows.

### **Initialising Agents**

To create an agent, you start by providing it with a set of functions or tools that clearly define its capabilities. Depending on the model, the agent will automatically use the function calling API (if it is available) or default to a standard ReAct agent loop.

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# initialize agent
agent = AgentWorkflow.from_tools_or_functions(
    [FunctionTool.from_defaults(multiply)],
    llm=llm
)
```

By default, **agents are stateless**, but they can be configured to remember past interactions by using a `Context` object, which is useful for chatbots that need to maintain conversational history. Additionally, agents in LlamaIndex are asynchronous and operate using Python's `await` operator.

```python
# stateless
response = await agent.run("What is 2 times 2?")

# remembering state
from llama_index.core.workflow import Context

ctx = Context(agent)

response = await agent.run("My name is Bob.", ctx=ctx)
response = await agent.run("What was my name again?", ctx=ctx)
```

### **Creating RAG Agents with QueryEngineTools**

**Agentic RAG** empowers agents to intelligently answer questions about your data. Instead of just returning document text, an agent can dynamically decide whether to use a retrieval tool or another flow entirely to formulate its answer. You can easily create this by wrapping an existing `QueryEngine` as a tool. When you do this, you must define a clear **name and description** so the LLM knows exactly how to correctly use the query engine.

```python
from llama_index.core.tools import QueryEngineTool

query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name",
    description="a specific description",
    return_direct=False,
)
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)
```

### **Creating Multi-agent systems**

LlamaIndex directly supports multi-agent systems using the **`AgentWorkflow`** class. In this setup, you give each specialized agent a name and description, and the system maintains a single active speaker. The agents have the ability to **hand off tasks to one another** based on their specialized capabilities. Narrowing the specific scope of each agent helps dramatically increase their general accuracy when responding to user requests. Furthermore, for highly complex setups, **agents can even be directly used as tools for other agents**.

```python
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.
# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

# Create and run the workflow
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

# Run the system
response = await agent.run(user_msg="Can you add 5 and 3?")
```

# **Creating Agentic Workflows in LlamaIndex**

A workflow in LlamaIndex provides a structured way to organize your code into sequential and manageable steps. Workflows are created by defining `Steps` that are triggered by `Events`, which in turn emit new `Events` to trigger further steps. This approach strikes a great balance between allowing agent autonomy and maintaining control over the overall system.

### **Key Benefits of Workflows**

Using workflows offers several main advantages:

- Clear organization of code into discrete steps.
- An event-driven architecture that allows for flexible control flow.
- Type-safe communication between steps.
- Built-in state management.
- Support for both simple and complex agent interactions.

```python
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

class MyWorkflow(Workflow):
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")

w = MyWorkflow(timeout=10, verbose=False)
result = await w.run()
```

### **Building Custom Workflows**

You can create manual workflows using several core mechanics:

- **Basic Steps & Events:** A single-step workflow is created by defining a class that inherits from `Workflow` and decorating its functions with `@step`. Special events like `StartEvent` and `StopEvent` indicate the beginning and end of the workflow.
- **Connecting Steps:** To link multiple steps, you create custom events that carry data between them. Type hinting is critical here to ensure the workflow executes correctly.

```python
from llama_index.core.workflow import Event

class ProcessingEvent(Event):
    intermediate_result: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent:
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(timeout=10, verbose=False)
result = await w.run()
result
```

- **Loops and Branches:** By using type hinting (such as the union operator `|`), you can create complex logic like branches, joins, or loops (e.g., using a `LoopEvent` as both an input and an output for a step).

```python
from llama_index.core.workflow import Event
import random

class ProcessingEvent(Event):
    intermediate_result: str

class LoopEvent(Event):
    loop_output: str

class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(verbose=False)
result = await w.run()
result
```

- **State Management:** If you need to keep track of the workflow's state so every step shares the same information, you can use a `Context` type hint on a step function parameter.
- **Drawing Workflows:** You can visualize your system by using the `draw_all_possible_flows` function, which saves a map of the workflow as an HTML file.

```python
from llama_index.utils.workflow import draw_all_possible_flows

w = ... # as defined in the previous section
draw_all_possible_flows(w, "flow.html")
```

### **Automating Workflows with Multi-Agent Systems**

Instead of manually creating every step, you can use the **`AgentWorkflow`** class to quickly build a multi-agent workflow. This class allows you to set up a system where multiple agents collaborate and hand off tasks to one another based on their specialized capabilities.

In an `AgentWorkflow`:

- One agent is designated as the **root agent** in the constructor, and it is the first to receive incoming user messages.
- Each agent evaluates the request and can either handle it directly using its tools, hand it off to a better-suited agent, or return a final response to the user.
- You can provide an initial state dictionary before starting the workflow, which is injected into a `state_prompt`. Agent tools can then read or modify this shared state as they work.

```python
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

# Create the workflow
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",
)

# Run the system
response = await workflow.run(user_msg="Can you add 5 and 3?")

```