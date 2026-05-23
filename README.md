# Intentmind 🧠

![PyPI version](https://img.shields.io/pypi/v/intentmind)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-Fair--Code-orange)

**Intentmind** is a **Cognitive Operating Layer for Persistent AI**. It acts as the foundational "Memory OS" for autonomous AI agents, copilots, simulators, and AI NPCs.

It moves beyond standard RAG by mimicking human cognitive processes—building a dynamic **Knowledge Graph**, forming associative connections, and employing biological "forgetting" mechanisms to maintain a pristine, highly relevant context window over months or years of interactions.

---

## 🛑 The Problem with Classic RAG

If you are building an autonomous system or an AI agent that lives for more than a single session, you will quickly hit the limits of standard Vector Database RAG wrappers:

1. **Amnesia by Proxy:** Standard RAG relies on overlapping embeddings. If a user talks about buying a *"car"* on Monday, and asks for help with *"insurance"* on Friday, standard RAG will fail to connect them unless the exact keywords or vector spaces strictly align.
2. **Context Bloat:** Vector DBs append data forever. Over time, your agent gets overwhelmed by thousands of trivial, outdated, or conflicting memories, destroying the LLM's accuracy.
3. **Flat Representation:** Standard RAG treats all data equally. It lacks the ability to understand that some memories are core to the user's identity, while others were just passing thoughts.

## 🌟 The Solution: Why Intentmind?

Intentmind doesn't just store documents; it builds a **living, breathing Cognitive Graph**. 

### 1. Associative Recall (Semantic Drift Recovery)
When data is ingested, Intentmind extracts core concepts (nodes) and connects them (edges). When a user mentions a concept, the system doesn't just do a vector search—it **traverses the graph**. 

**Example of Semantic Drift:**
User talks about their *car* ➔ then plans a *trip* ➔ then asks about *insurance*. 
Intentmind naturally traverses the graph from `car` to `insurance`, pulling in the exact historical context your agent needs to sound truly intelligent, even if the raw embeddings don't directly match.

### 2. Biological Forgetting (Energy & Decay)
Intentmind introduces an `Energy` system. Every time a memory is recalled, its connection strengthens (like human synapses). Unused, irrelevant, or noisy memories slowly decay and are eventually "forgotten" (archived). This prevents context bloat and ensures your agent only remembers what actually matters.

### 3. Cognitive State Modulation
Previously known as the "Emotion Engine", this system features an **Adaptive Urgency Matrix**. It dynamically scores the user's current query for stress, urgency, or sentiment, and applies **retrieval priority modulation**. A highly urgent query immediately surfaces hard, high-confidence facts, while a casual conversational query triggers broader, associative exploration.

---

## ⚔️ Why not just GraphRAG?

GraphRAG is fantastic for static document analysis, but it fails for **persistent, living agents**. 

| Feature | Vanilla RAG | GraphRAG | Intentmind |
| :--- | :--- | :--- | :--- |
| **Primary Use Case** | Q&A on Docs | Knowledge Discovery | **Living AI Agents** |
| **Memory Architecture** | Static Vectors | Static Graph | **Dynamic Fluid Graph** |
| **Associative Activation** | No | Hardcoded Edges | **Energy Propagation** |
| **Memory Aging** | No (Bloats forever) | No | **Biological Forgetting (Decay)** |
| **Reinforcement** | No | No | **Synaptic Strengthening via Reuse** |

## 📊 Performance Benchmarks

Intentmind is built to survive long-term deployments where other systems degrade.

| Benchmark Metric | Vanilla RAG | GraphRAG | Intentmind |
| :--- | :--- | :--- | :--- |
| **Delayed Recall Accuracy** | Low | Medium | **High** |
| **Semantic Drift Resistance** | Poor | Medium | **Excellent** |
| **Memory Pollution (Noise) Resistance** | Fails at scale | Medium | **High** (Auto-pruning) |
| **Long-session Degradation** | Severe | Moderate | **Near-Zero** |
| **Contradiction Resolution** | Low | Low | **High** (Energy-based priority) |

---

## 🎯 Who is this for?

- **Autonomous Agents & AI Personas:** Give your AI agents long-term, evolving memory. As they interact with the world, their internal cognitive graph adapts, making them highly personalized and context-aware.
- **Persistent Copilots:** Build customer support bots, coding assistants, or personal companions that remember user preferences seamlessly over years of interaction without polluting the LLM context window.
- **Complex Simulators:** Drive simulations where multiple entities negotiate, learn, and forget based on cognitive connections.

---

## 🚀 Quick Start

### Installation

```bash
# Core package
pip install intentmind

# With recommended production dependencies (FAISS, SentenceTransformers, OpenAI)
pip install "intentmind[all]"
```

### 1-Minute Integration

Intentmind is completely language-agnostic.

```python
from intentmind import IntentmindMemory
from intentmind.embeddings import SentenceTransformerEmbedder

# 1. Initialize the Cognitive Engine
mem = IntentmindMemory(embedder=SentenceTransformerEmbedder())

# 2. Teach it concepts (Happens naturally during conversation)
mem.add("I need car insurance but I have no money.")
mem.add("I bought a new car and will drive to London.")

# 3. Ask a tangential question
result = mem.query("I have a car and I am going to London.")

# 4. Watch it traverse the graph and retrieve the exact context
for item in result["memories"]["items"]:
    print(f"[{item['layer']}] Path: {item.get('path', [])} -> {item['text']}")
```

---

## 🔌 Seamless Integrations

### LangChain Adapter
Drop Intentmind straight into your existing LangChain pipelines as a custom, ultra-smart Retriever.
```python
from intentmind.integrations.langchain import IntentmindRetriever

retriever = IntentmindRetriever(memory=mem)
docs = retriever.invoke("How does the energy model work?")
```

### Persistence (Save & Load)
Save the entire brain (vectors, nodes, edges, decay states) to disk instantly.
```python
mem.save("my_agent_brain.json")
mem = IntentmindMemory.load("my_agent_brain.json")
```

---

## 📊 Visualizing the "Brain"

Intentmind isn't a black box. You can export the entire Cognitive Graph to an interactive HTML map to literally see what your AI is thinking and how it connects concepts.

```python
mem.visualize("memory_map.html")
```
*(Open `memory_map.html` in your browser to explore clustering, energy levels, and synaptic connections)*

---

## 🌐 Demo API & React UI

Intentmind includes a built-in FastAPI backend and a beautiful React/Vite UI to test your graphs visually.

1. **Start Backend:**
   ```bash
   pip install "intentmind[api]"
   # Make sure OPENAI_API_KEY is in your .env
   python api.py
   ```
2. **Start Frontend UI:**
   ```bash
   cd ui
   npm install
   npm run dev
   ```

---

## 🤝 Contributing & License

Intentmind is released under the **Intentmind Fair-Code License**. 

- **Personal & Academic Use:** Completely free! You can use, modify, and experiment with Intentmind for your personal projects, academic research, or hobby apps.
- **Commercial Use:** If you plan to use Intentmind in a commercial product, SaaS, or any revenue-generating service, you **must** obtain prior written consent or a commercial license. 

For commercial licensing inquiries, please reach out via GitHub issues or contact the author directly.

We welcome contributions for new embedders, LLM adapters, and vector store integrations! Build the future of AI memory with us.
