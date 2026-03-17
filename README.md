# 🤖 Hugging Face AI Agents Course

This repository documents my journey through the [Hugging Face AI Agents Course](https://huggingface.co/learn/agents-course). This comprehensive program covers the theory, design, and implementation of autonomous AI agents using state-of-the-art open-source tools.

## 🌟 Course Overview

The course is designed to take learners from the basics of Large Language Models (LLMs) to building sophisticated, multi-agent systems capable of solving real-world tasks.

### 📚 Curriculum Structure

#### **Unit 1: Agent Fundamentals**
* **The Brain:** Understanding LLMs, Messages, Special Tokens, and Chat Templates.
* **The Workflow:** Mastering the **Thought → Action → Observation** cycle.
* **Reasoning:** Exploring ReAct and Chain-of-Thought (CoT) prompting strategies.
* **First Agent:** Building "Alfred," a basic agent using the `smolagents` library.

#### **Unit 2: Frameworks for AI Agents**
Deep dives into the industry-standard libraries for orchestrating agents:
* **smolagents:** Hugging Face's lightweight, code-first agent library.
* **LlamaIndex:** Building context-augmented agents for production.
* **LangGraph:** Mastering stateful multi-agent orchestration.

#### **Unit 3: Use Cases & Agentic RAG**
* Implementing **Agentic RAG** (Retrieval Augmented Generation) where agents decide when and how to search for information.
* Building specialized agents for web searching, image generation, and API interaction.

#### **Unit 4: Final Project & Benchmarking**
* **The GAIA Benchmark:** Testing agent performance against complex, real-world tasks.
* **Final Certification:** Designing, testing, and deploying a custom agent to the Hugging Face Hub.

---

## 🛠️ Tech Stack

* **Libraries:** `transformers`, `smolagents`, `langgraph`, `llama-index`
* **Models:** SmolLM2, Llama 3.1, Mistral
* **Deployment:** Hugging Face Spaces & Model Hub

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/huggingface-agents-journey.git](https://github.com/YOUR_USERNAME/huggingface-agents-journey.git)
   ```
2. **Install core dependencies**
    ```bash
    pip install smolagents transformers huggingface_hub
    ```
3. **Configure Environment**
Ensure your HF_TOKEN is set to access models from the Hugging Face Hub.
   

