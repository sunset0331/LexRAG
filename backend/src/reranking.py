from typing import List, Dict
from sentence_transformers import CrossEncoder

class DocumentReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name, max_length=512)
        
    def rerank(self, query: str, documents: List[Dict], top_k: int = 3) -> List[Dict]:
        """
        Reranks a list of documents using a Cross-Encoder.
        """
        if not documents:
            return []
            
        print(f"Reranking {len(documents)} documents...")
        
        # Prepare pairs for the Cross-Encoder: (query, document_text)
        pairs = [[query, doc["content"]] for doc in documents]
        
        # Get scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for idx, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[idx])
            
        # Sort by rerank score descending
        reranked_docs = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        
        return reranked_docs[:top_k]
