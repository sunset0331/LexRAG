import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from pypdf import PdfReader

from data.mock_contracts import MOCK_CONTRACTS
from src.rag_pipeline import LegalRAGPipeline

app = FastAPI(title="Legal RAG API")

# Setup CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize and hydrate pipeline
pipeline = LegalRAGPipeline()
pipeline.ingest_data(MOCK_CONTRACTS)

class QueryRequest(BaseModel):
    query: str
    thread_id: str

class Source(BaseModel):
    id: str
    title: str
    content: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    intent_used: str
    sources: List[Source]
    chat_history: Optional[List[dict]] = None

@app.post("/api/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        result = pipeline.process_query(request.query, request.thread_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def get_sessions():
    return pipeline.get_sessions()

@app.get("/api/sessions/{thread_id}")
async def get_session_history(thread_id: str):
    history = pipeline.get_session_history(thread_id)
    return {"history": history}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    try:
        reader = PdfReader(file.file)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
            
        doc = {
            "id": file.filename,
            "title": file.filename,
            "parties": "User Upload",
            "content": text
        }
        
        # Ingest to FAISS and BM25
        pipeline.indexer.ingest([doc])
        
        # Add to MOCK_CONTRACTS for UI listing
        MOCK_CONTRACTS.append(doc)
        
        return {"message": f"Successfully indexed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents():
    """Returns the list of ingested documents for frontend display."""
    return [{"id": d["id"], "title": d["title"], "parties": d["parties"]} for d in MOCK_CONTRACTS]

if __name__ == "__main__":
    import uvicorn
    # Start the server (reload=False to prevent file writes from triggering a reload loop)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
