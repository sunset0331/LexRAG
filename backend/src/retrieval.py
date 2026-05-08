from typing import List, Dict
import os
import uuid

class HybridRetriever:
    def __init__(self, indexer):
        self.indexer = indexer
        
    def retrieve(self, query: str, intent: str, top_k: int = 10) -> List[Dict]:
        if not self.indexer.vector_store or not self.indexer.bm25_index:
            print("Warning: Indices not built yet.")
            return []
            
        # Adjust weights based on routing intent
        if intent == "keyword_heavy":
            dense_weight, sparse_weight = 0.2, 0.8
        elif intent == "semantic_heavy":
            dense_weight, sparse_weight = 0.8, 0.2
        else: # balanced
            dense_weight, sparse_weight = 0.5, 0.5
            
        print(f"Applying RRF with intent: {intent}")
            
        # 1. Dense Retrieval (FAISS)
        print("Executing Vector Search...")
        # FAISS returns (Document, score) where score is L2 distance (lower is better)
        # But similarity_search handles it, we can just use similarity_search
        dense_results = self.indexer.vector_store.similarity_search(query, k=top_k)
        
        dense_ranks = {}
        for rank, doc in enumerate(dense_results):
            # doc.metadata contains our chunk info, but we need the chunk ID
            # FAISS similarity_search doesn't return the ID natively in all versions
            # So we match by content
            content = doc.page_content
            # Find the ID in documents_metadata
            doc_id = None
            for meta_doc in self.indexer.documents_metadata:
                if meta_doc["content"] == content:
                    doc_id = meta_doc["id"]
                    break
            
            if doc_id:
                dense_ranks[doc_id] = rank + 1
                
        # 2. Sparse Retrieval (BM25)
        print("Executing BM25 Search...")
        tokenized_query = query.lower().split()
        bm25_scores = self.indexer.bm25_index.get_scores(tokenized_query)
        
        # Get top K indices
        top_n = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        sparse_ranks = {}
        for rank, idx in enumerate(top_n):
            doc_id = self.indexer.documents_metadata[idx]["id"]
            sparse_ranks[doc_id] = rank + 1
            
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        k_constant = 60
        
        all_ids = set(dense_ranks.keys()).union(set(sparse_ranks.keys()))
        for doc_id in all_ids:
            dense_rank = dense_ranks.get(doc_id, float('inf'))
            sparse_rank = sparse_ranks.get(doc_id, float('inf'))
            
            dense_score = 0 if dense_rank == float('inf') else 1.0 / (k_constant + dense_rank)
            sparse_score = 0 if sparse_rank == float('inf') else 1.0 / (k_constant + sparse_rank)
            
            final_score = (dense_weight * dense_score) + (sparse_weight * sparse_score)
            rrf_scores[doc_id] = final_score
            
        # Sort by final RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Fetch the actual documents
        final_results = []
        for doc_id in sorted_ids[:top_k]:
            for meta_doc in self.indexer.documents_metadata:
                if meta_doc["id"] == doc_id:
                    result = dict(meta_doc)
                    result["rrf_score"] = rrf_scores[doc_id]
                    final_results.append(result)
                    break
                    
        return final_results
