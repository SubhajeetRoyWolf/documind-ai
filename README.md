# 🧠 DocuMind AI
### Agentic RAG Document Intelligence System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Azure OpenAI](https://img.shields.io/badge/Azure-OpenAI-0078D4)
![Pinecone](https://img.shields.io/badge/Vector_DB-Pinecone-green)
![LangGraph](https://img.shields.io/badge/Agents-LangGraph-purple)
![CI](https://github.com/SubhajeetRoyWolf/documind-ai/actions/workflows/eval_ci.yml/badge.svg)

A production-grade, multi-agent document intelligence system that answers natural language questions about any PDF using RAG, LangGraph agent orchestration, and Azure OpenAI.

---

## 🏗️ Architecture

```
User Query
    ↓
🧠 Planner Agent (LangGraph) — decides strategy
    ↓
🔍 Retriever Agent — Pinecone vector search + Cohere rerank
    ↓
✍️ Generator Agent — GPT-4o grounded generation
    ↓
🔎 Critic Agent — hallucination detection + approval
    ↓
Cited Answer
```

---

## ✨ Features

- **Multi-agent orchestration** — LangGraph StateGraph with Planner, Retriever, Generator, Critic agents
- **Production RAG pipeline** — PDF ingestion → chunking → Azure OpenAI embeddings → Pinecone → Cohere reranking
- **Hallucination detection** — NLI-based faithfulness scoring
- **Eval harness** — RAGAS metrics + pytest regression suite with CI/CD gating
- **LoRA fine-tuning** — PEFT adapter training (0.05% params)
- **MLflow tracking** — eval metrics across experiments
- **FastAPI microservice** — /ingest, /query, /health endpoints
- **Streamlit UI** — interactive chat interface

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Azure OpenAI GPT-4o |
| Embeddings | text-embedding-3-small |
| Vector DB | Pinecone (serverless) |
| Reranker | Cohere rerank-english-v3.0 |
| Agent framework | LangGraph + LlamaIndex |
| Eval framework | RAGAS + pytest |
| Experiment tracking | MLflow |
| Fine-tuning | LoRA/PEFT (TinyLlama) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Infra | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## 🚀 Quick Start

### 1. Clone and setup
```bash
git clone https://github.com/SubhajeetRoyWolf/documind-ai
cd documind-ai
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in your API keys
```

### 3. Run with Docker
```bash
docker-compose up --build
```

### 4. Or run locally
```bash
# Start FastAPI
cd src && uvicorn main:app --reload --port 8000

# Start Streamlit (new terminal)
streamlit run src/streamlit_app.py
```

---

## 📊 Eval Results

| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Faithfulness | 0.875 | 0.70 | ✅ |
| Answer Relevancy | 0.875 | 0.70 | ✅ |
| Keyword Match | 0.792 | 0.50 | ✅ |
| Pass Rate | 100% | 70% | ✅ |

---

## 📁 Project Structure

```
documind-ai/
├── src/
│   ├── main.py              # FastAPI microservice
│   ├── ingestion.py         # PDF → Pinecone pipeline
│   ├── retrieval.py         # Query → GPT-4o pipeline
│   ├── agent_state.py       # LangGraph shared state
│   ├── agents.py            # Planner/Retriever/Generator/Critic
│   ├── graph_v2.py          # LangGraph orchestration
│   ├── memory.py            # Multi-turn conversation memory
│   ├── web_search_tool.py   # LlamaIndex FunctionTool
│   ├── ragas_eval.py        # RAGAS evaluation harness
│   ├── lora_finetune.py     # LoRA/PEFT fine-tuning
│   ├── streamlit_app.py     # Chat UI
│   └── test_rag_eval.py     # pytest regression suite
├── .github/workflows/
│   └── eval_ci.yml          # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🔑 Environment Variables

```env
AZURE_OPENAI_KEY=your_key
AZURE_OPENAI_ENDPOINT=https://xxx.openai.azure.com/
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o
AZURE_OPENAI_EMBED_DEPLOYMENT=text-embedding-3-small
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=documind-index
COHERE_API_KEY=your_key
```

## 🧠 Key Concepts Demonstrated

- **RAG** — Retrieval Augmented Generation pipeline
- **Agent orchestration** — Multi-agent LangGraph system
- **Vector search** — Pinecone ANN + Cohere reranking
- **Hallucination detection** — Critic agent + RAGAS faithfulness
- **LoRA fine-tuning** — PEFT parameter-efficient training
- **MLflow** — Experiment tracking and model registry
- **CI/CD** — Automated eval gating on every push

