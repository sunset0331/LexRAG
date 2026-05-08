from .indexing import DocumentIndexer
from .routing import QueryRouter
from .retrieval import HybridRetriever
from .reranking import DocumentReranker
from .generation import AnswerGenerator
import os
import sqlite3
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

class GraphState(TypedDict):
    query: str
    chat_history: List[Dict]
    intent: str
    documents: List[Dict]
    answer: str
    sources: List[Dict]

class LegalRAGPipeline:
    def __init__(self):
        print("Initializing Legal RAG Pipeline with LangGraph...")
        self.indexer = DocumentIndexer()
        self.router = QueryRouter()
        self.retriever = HybridRetriever(self.indexer)
        self.reranker = DocumentReranker()
        self.generator = AnswerGenerator()
        
        # Setup DB for checkpointer
        os.makedirs("data", exist_ok=True)
        self.conn = sqlite3.connect("data/checkpoints.sqlite", check_same_thread=False)
        self.memory = SqliteSaver(self.conn)
        
        # Build Graph
        workflow = StateGraph(GraphState)
        
        workflow.add_node("route", self.route_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rerank", self.rerank_node)
        workflow.add_node("generate", self.generate_node)
        
        workflow.add_edge(START, "route")
        workflow.add_edge("route", "retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)
        
        self.app = workflow.compile(checkpointer=self.memory)

    def route_node(self, state: GraphState):
        intent = self.router.route_query(state["query"])
        return {"intent": intent}
        
    def retrieve_node(self, state: GraphState):
        docs = self.retriever.retrieve(state["query"], state.get("intent", "balanced"), top_k=10)
        return {"documents": docs}
        
    def rerank_node(self, state: GraphState):
        best_docs = self.reranker.rerank(state["query"], state.get("documents", []), top_k=3)
        return {"documents": best_docs}
        
    def generate_node(self, state: GraphState):
        result = self.generator.generate(state["query"], state.get("documents", []), chat_history=state.get("chat_history", []))
        
        new_history = state.get("chat_history", [])
        if new_history is None:
            new_history = []
            
        new_history = new_history + [
            {"role": "user", "content": state["query"]},
            {"role": "bot", "content": result["answer"], "intent": state.get("intent", "balanced"), "sources": result["sources"]}
        ]
        
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "chat_history": new_history
        }

    def ingest_data(self, documents):
        print("Starting data ingestion...")
        self.indexer.ingest(documents)
        print("Data ingestion complete.")
        
    def process_query(self, query: str, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"query": query}
        
        print(f"\n--- Processing Query via Graph: '{query}' [Thread: {thread_id}] ---")
        result = self.app.invoke(inputs, config=config)
        
        return {
            "answer": result["answer"],
            "intent_used": result.get("intent", "balanced"),
            "sources": result.get("sources", []),
            "chat_history": result.get("chat_history", [])
        }

    def get_sessions(self):
        try:
            # LangGraph SqliteSaver stores checkpoints in 'checkpoints' table
            # thread_id is a column
            cursor = self.conn.cursor()
            # Get distinct thread_ids. SqliteSaver stores thread_id, checkpoint_ns, checkpoint_id
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
            rows = cursor.fetchall()
            sessions = []
            for row in rows:
                thread_id = row[0]
                state = self.app.get_state({"configurable": {"thread_id": thread_id}}).values
                history = state.get("chat_history", [])
                if history and len(history) > 0:
                    title = history[0]["content"]
                    sessions.append({
                        "thread_id": thread_id, 
                        "title": title[:30] + "..." if len(title) > 30 else title
                    })
            # Reverse to show newest first
            return list(reversed(sessions))
        except Exception as e:
            print("Error fetching sessions:", e)
            return []
            
    def get_session_history(self, thread_id: str):
        try:
            state = self.app.get_state({"configurable": {"thread_id": thread_id}}).values
            return state.get("chat_history", [])
        except Exception:
            return []
