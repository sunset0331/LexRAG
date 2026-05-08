import os
import uuid
from typing import List, Dict, Any

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentIndexer:
    def __init__(self, persist_directory: str = "./faiss_index"):
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = persist_directory
        self.vector_store = None
        self.bm25_index = None
        self.documents_metadata = []
        self.corpus = []
        
    def ingest(self, documents: List[Dict[str, str]]):
        """
        documents: list of dicts with 'id', 'title', 'parties', 'content'
        """
        # 1. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunked_docs = []
        metadatas = []
        ids = []
        
        for doc in documents:
            chunks = text_splitter.split_text(doc["content"])
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                chunked_docs.append(chunk)
                
                meta = {
                    "source_id": doc["id"],
                    "title": doc["title"],
                    "parties": doc["parties"],
                    "chunk_index": i
                }
                metadatas.append(meta)
                ids.append(chunk_id)
                
                # Keep original data for BM25 mapping
                self.documents_metadata.append({
                    "id": chunk_id,
                    "content": chunk,
                    "metadata": meta
                })
        
        # 2. Build Dense Index (FAISS)
        if self.vector_store is None:
            self.vector_store = FAISS.from_texts(
                texts=chunked_docs,
                embedding=self.embeddings,
                metadatas=metadatas,
                ids=ids
            )
        else:
            self.vector_store.add_texts(
                texts=chunked_docs,
                metadatas=metadatas,
                ids=ids
            )
            
        # Optional: Save FAISS to disk
        # os.makedirs(self.persist_directory, exist_ok=True)
        # self.vector_store.save_local(self.persist_directory)
        
        # 3. Build Sparse Index (BM25)
        # Tokenize by simply splitting on whitespace for BM25
        new_corpus = [chunk.lower().split() for chunk in chunked_docs]
        self.corpus.extend(new_corpus)
        self.bm25_index = BM25Okapi(self.corpus)
        
        print(f"Indexed {len(chunked_docs)} chunks from {len(documents)} documents.")
