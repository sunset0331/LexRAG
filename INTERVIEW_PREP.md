# LexRAG: Advanced Hybrid Legal RAG System
**Interview Preparation Report & Architecture Walkthrough**

---

## 1. Executive Summary
LexRAG is a production-ready, full-stack Retrieval-Augmented Generation (RAG) application specifically engineered to analyze and query complex legal documents (like NDAs, Employment Agreements, and custom user-uploaded PDFs). 

Unlike basic "vanilla" RAG systems that rely solely on simple semantic search, LexRAG implements an **Advanced Hybrid Retrieval Pipeline**. It combines Dense (Semantic) and Sparse (Keyword) search, merges them using Reciprocal Rank Fusion (RRF), re-scores the results using a Cross-Encoder, and orchestrates the entire flow using a stateful LangGraph architecture with local SQLite persistence.

---

## 2. The Tech Stack
If asked "What is the tech stack and why did you choose it?", here is the breakdown:

### **Backend:**
*   **FastAPI:** Chosen for its blazing fast performance, native async support, and automatic OpenAPI documentation.
*   **LangGraph:** Replaced linear LangChain scripts. Chosen because it treats the RAG pipeline as a State Machine, allowing for complex cyclic workflows, easy debugging, and native memory checkpointing.
*   **Vector Database (Dense):** **FAISS** (Facebook AI Similarity Search). Chosen over ChromaDB because we encountered file-locking issues with Chroma during concurrent dev server reloads. FAISS runs blazingly fast in-memory.
*   **Sparse Database:** **Okapi BM25**. Chosen because legal documents require exact keyword matching (e.g., "Section 4.2b", "Acme Corp"). Semantic search often fails at exact string matching.
*   **Embeddings Model:** `all-MiniLM-L6-v2` (HuggingFace). A small, fast sentence-transformer perfect for local embedding.
*   **Cross-Encoder Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`. Chosen to precisely score the final candidates retrieved by the hybrid search.
*   **Generative LLM:** `Qwen/Qwen2.5-72B-Instruct` (via HuggingFace Inference API). Chosen after migrating away from rate-limited Gemini APIs. Qwen 72B is an open-weights powerhouse that excels at following complex legal instructions without hallucinating.

### **Frontend:**
*   **React + Vite:** Chosen for fast hot-module reloading and a snappy Single Page Application (SPA) experience.
*   **Vanilla CSS:** Custom-built glassmorphism and modern UI components.

---

## 3. Architecture & Data Flow (The "How It Works" Pitch)

When asked to "Walk me through the architecture," explain this exact flow:

### Phase 1: Ingestion & Indexing
1.  **PDF Upload:** The user uploads a PDF via the React UI.
2.  **Chunking:** The backend uses `pypdf` to extract text, and `RecursiveCharacterTextSplitter` to chunk the text into 500-token blocks with a 50-token overlap to preserve context across boundaries.
3.  **Dual Indexing:** Every chunk is simultaneously embedded into a high-dimensional vector and stored in **FAISS**, AND tokenized into a keyword corpus stored in **BM25**.

### Phase 2: The LangGraph Query Pipeline
When a user asks a question, the LangGraph State Machine spins up:
1.  **`Route Node`:** The query is sent to the LLM router which categorizes the user's intent into `keyword_heavy` (e.g., "Find the clause about termination"), `semantic_heavy` (e.g., "What is the general tone of the liability section?"), or `balanced`.
2.  **`Retrieve Node` (Hybrid Search + RRF):** 
    *   The query searches both FAISS (Dense) and BM25 (Sparse).
    *   It uses **Reciprocal Rank Fusion (RRF)** to algorithmically merge the two ranked lists. If the router determined the query was `keyword_heavy`, the BM25 scores get a higher weight in the RRF formula. It selects the Top 10 combined chunks.
3.  **`Rerank Node`:** Bi-encoders (FAISS) are fast but imprecise. We take those Top 10 chunks and pass them through a **Cross-Encoder**. The Cross-Encoder reads the Query and the Chunk *together* simultaneously, generating a highly accurate relevance score. We keep the absolute best Top 3 chunks.
4.  **`Generate Node`:** The Top 3 chunks, along with the user's query and the *Chat History*, are injected into a strict prompt template. Qwen2.5 generates the final answer, explicitly citing the source documents.
5.  **`Memory Checkpoint`:** LangGraph's `SqliteSaver` automatically intercepts the final state and saves the conversation turn to `checkpoints.sqlite` under a unique `thread_id`.

---

## 4. Key Design Decisions & Trade-offs (Crucial for Interviews)

### **Why use a Cross-Encoder if you already have FAISS?**
*   **The Trade-off:** Speed vs Precision.
*   **The Answer:** "FAISS uses Bi-Encoders, which embed the query and the document separately. It's fast enough to search millions of documents instantly, but it misses nuance. Cross-Encoders process the query and document *together* through the neural network, which is highly accurate but computationally expensive. By using FAISS to narrow down millions of docs to just 10, and then using a Cross-Encoder to re-rank only those 10, I get the speed of Bi-Encoders with the precision of Cross-Encoders."

### **Why use Reciprocal Rank Fusion (RRF)?**
*   **The Answer:** "Dense search (FAISS) is great for 'meaning' (e.g., matching 'dog' with 'canine'). Sparse search (BM25) is great for exact keywords (e.g., finding ID numbers or specific names). RRF is a mathematical formula that takes the rankings from both systems and merges them without needing to normalize their wildly different scoring metrics. It ensures that a document that ranks moderately well in both systems bubbles up to the top."

### **Why migrate from ChromaDB to FAISS?**
*   **The Answer:** "ChromaDB is fantastic, but during local development with FastAPI's hot-reloading, I encountered SQLite file-locking issues (`database is locked`) because Chroma constantly writes to disk. I swapped to FAISS because it operates entirely in-memory, which completely eliminated the I/O bottleneck and significantly sped up retrieval times during development."

### **Why use LangGraph instead of standard LangChain?**
*   **The Answer:** "Standard LangChain uses `Runnables` which form rigid, linear chains. LangGraph treats the workflow as a graph/state-machine. This allowed me to easily implement the `SqliteSaver` checkpointer, which automatically intercepts the state between nodes and persists the chat history to a database. It allows the React frontend to fetch past sessions via `thread_id` without me having to write complex database management code."

---

## 5. Potential Interview Questions

**Q1: How do you handle hallucinations in your legal RAG?**
> **A:** "I use strict prompt engineering telling the LLM to *only* use the provided context and to state 'I don't know' if it's missing. More importantly, the Cross-Encoder re-ranking ensures that the context injected into the prompt is highly relevant. Finally, I force the LLM to output citations (Document Title and ID) so the user can verify the source."

**Q2: If the document corpus grew to 10 million PDFs, how would you scale this?**
> **A:** "I would move the vector storage out of local FAISS and into a managed cloud provider like Pinecone or Milvus. I would also move the LangGraph checkpointer out of local SQLite and into PostgreSQL. For ingestion, I would set up a queue (like Celery/Redis) to handle the PDF chunking and embedding asynchronously so the main API doesn't block."

**Q3: How does conversational memory work in your system?**
> **A:** "The frontend generates a unique `thread_id` for every new chat. Every time the user sends a message, that `thread_id` is passed to the FastAPI backend. LangGraph's checkpointer intercepts the execution, looks up the `thread_id` in the local SQLite database, loads the past messages into the Graph's State, executes the generation, and then saves the new turn back to the database."
