# Warring States KG : RAG Demo

This project implements a **Retrieval-Augmented Generation (RAG)** system based on a **Knowledge Graph (KG)** and **Knowledge Graph Embeddings (KGE)**.

It allows you to ask questions about the *Warring States period* of ancient China and compare:

*  A standard LLM answer
* A KG-enhanced RAG answer

---

## Features

* Knowledge Graph built from historical data
* KGE-based similarity search
* RAG pipeline combining KG + LLM
* Interactive UI with Gradio
* Evaluation module

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Valdg2165/RAG_Warring_states_period.git
cd RAG_Warring_states_period
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama (for LLM)

Make sure you have Ollama installed and running:

```bash
ollama run llama3
```

---

## ▶️ Usage

Run the Gradio app:

```bash
python src/rag/app.py
```

Then open:

```
http://localhost:7860
```

---

## Example Questions

* Who were the students of Confucius?
* Which states were at war with Qin?
* Who did Aristotle influence?

---

## How it works

1. Detect entities in the question
2. Expand with KGE similar entities
3. Retrieve triples from the knowledge graph
4. Inject context into the LLM
5. Generate answer

---

## Evaluation

You can run the built-in evaluation directly in the UI.

It compares:

* Baseline LLM
* RAG (KG + KGE)

---

## Project Structure

```
src/
  rag/        # RAG pipeline
  kg/         # Knowledge graph construction
  kge/        # Embeddings training
  reason/     # Reasoning (SWRL)

data/
kg_artifacts/
```

---

## Tech Stack

* Python
* Gradio
* RDF / SPARQL
* Knowledge Graph Embeddings
* Ollama (LLMs)