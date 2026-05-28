# Multi-Agent RAG Assistant

A LangGraph-powered multi-agent RAG (Retrieval-Augmented Generation) system that allows users to upload multiple PDF documents and ask natural language questions. The application retrieves relevant document chunks using semantic search and generates accurate, source-cited answers using Groq LLMs.

---

## Features

* Multi-PDF Upload Support
* Semantic Search using FAISS
* LangGraph StateGraph Workflow
* Source-Cited Responses
* Multi-turn Chat Interface
* Streamlit Frontend
* HuggingFace Embeddings
* Groq LLM Integration
* Source Snippet Display

---

## Tech Stack

* Python
* Streamlit
* LangChain
* LangGraph
* FAISS
* HuggingFace Embeddings
* Groq API
* PyMuPDF

---

## Workflow

PDF Upload
→ Document Parsing
→ Chunking
→ Embedding Generation
→ FAISS Vector Store
→ Retrieval Agent
→ Response Agent
→ Source-Cited Answer

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <your_repo_link>
cd <repo_name>
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Virtual Environment

#### Windows

```bash
.venv\Scripts\activate
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Create `.env` File

Add your Groq API key:

```env
GROQ_API_KEY=your_api_key
```

### 6. Run Application

```bash
streamlit run app.py
```

---

## Project Architecture

### Retrieval Agent

Retrieves relevant document chunks from FAISS vector database.

### Response Agent

Generates grounded answers using retrieved context.

### LangGraph Orchestration

Uses StateGraph to orchestrate retrieval and response agents.

---

## Future Improvements

* Hybrid Search (BM25 + Vector)
* Evaluator Agent
* Streaming Responses
* Docker Deployment
* Persistent Memory
