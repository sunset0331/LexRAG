# Advanced Hybrid Legal RAG System

A production-ready Retrieval-Augmented Generation (RAG) application specifically tailored for querying complex Legal Documents (like NDAs, Employment Agreements, and custom PDFs). 

This system uses a highly optimized **Hybrid Search (Dense + Sparse)** architecture combined with **Reciprocal Rank Fusion (RRF)**, **Cross-Encoder Re-ranking**, and a **Stateful LangGraph Pipeline** to deliver incredibly accurate and reliable legal answers.

##  Key Features

*   **Hybrid Search Architecture:** Combines semantic search (FAISS Vector Database) for conceptual queries with exact-keyword sparse search (Okapi BM25) for finding specific clauses (e.g., "Section 4.2").
*   **Reciprocal Rank Fusion (RRF):** Dynamically merges the scores from both dense and sparse retrievers based on the user's intent.
*   **Dynamic Intent Routing:** An LLM router categorizes queries as `keyword_heavy`, `semantic_heavy`, or `balanced` to intelligently weight the RRF search.
*   **Cross-Encoder Re-ranking:** A local `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the retrieved chunks for maximum precision before passing them to the generative model.
*   **Stateful LangGraph Memory:** Chat history is persisted automatically into a local SQLite database (`SqliteSaver`) allowing for multi-turn conversations and a persistent Chat Sidebar.
*   **PDF Upload Support:** Users can seamlessly upload custom `.pdf` files on the fly, which are automatically chunked, embedded, and added to the FAISS and BM25 indices incrementally.
*   **LangSmith Observability:** Native support for LangSmith tracing to debug and monitor the LangGraph node executions.

## Technology Stack

*   **Backend:** FastAPI, LangChain, LangGraph, PyPDF
*   **Frontend:** React, Vite, Lucide-React
*   **Vector Database (Dense):** FAISS (running locally)
*   **Sparse Database:** rank_bm25
*   **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace)
*   **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
*   **LLM Generator & Router:** `Qwen/Qwen2.5-72B-Instruct` (via HuggingFace Serverless Inference API)

##  Setup & Installation

### 1. Backend Setup

Open a terminal and navigate to the backend directory:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install fastapi uvicorn pydantic python-dotenv langchain-huggingface langchain-community langchain-core rank-bm25 pypdf sentence-transformers faiss-cpu langgraph langgraph-checkpoint-sqlite
```

Configure your environment variables by creating or editing the `.env` file in the `backend/` directory:

```env
# Required: HuggingFace API Token (Free Tier works perfectly)
HUGGINGFACEHUB_API_TOKEN="your_huggingface_token_here"

# Optional: Enable LangSmith Tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="your_langsmith_key_here"
LANGCHAIN_PROJECT="Legal_RAG_Project"
```

Start the backend server:

```bash
python main.py
```
*Note: The backend runs on `http://localhost:8000`. We run it without Uvicorn auto-reload to prevent infinite loops when FAISS or SQLite writes to disk.*

### 2. Frontend Setup

Open a new terminal and navigate to the frontend directory:

```bash
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
*The frontend will run on `http://localhost:5173`.*

##  How it Works (LangGraph Workflow)

1. **`Route Node`:** The user's query is sent to Qwen 72B to classify it as keyword-heavy, semantic-heavy, or balanced.
2. **`Retrieve Node`:** Performs similarity search in FAISS and BM25, merges them using RRF, and retrieves the Top 10 candidate chunks.
3. **`Rerank Node`:** Passes the query and Top 10 chunks through a Cross-Encoder to get the absolute best Top 3 most relevant chunks.
4. **`Generate Node`:** Synthesizes the final response using Qwen 72B, citing the exact documents used, and appends the turn to the SQLite persistent chat history.
