import os
from typing import List, Dict
from huggingface_hub import InferenceClient

class AnswerGenerator:
    def __init__(self):
        self.api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not self.api_token:
            print("WARNING: HUGGINGFACEHUB_API_TOKEN not found in environment. Generator will not work.")
            self.client = None
        else:
            print("Initializing HuggingFace Generator (Qwen)...")
            self.client = InferenceClient(token=self.api_token)
            self.model = "Qwen/Qwen2.5-72B-Instruct"
            
    def generate(self, query: str, context_docs: List[Dict], chat_history: list = None) -> Dict:
        if not self.client:
            return {"answer": "Error: HuggingFace API token not configured.", "sources": []}
            
        if not context_docs:
            return {"answer": "I could not find any relevant information in the provided documents to answer your question.", "sources": []}
            
        # Build the context string
        context_string = ""
        sources = []
        for idx, doc in enumerate(context_docs):
            title = doc["metadata"].get("title", "Unknown Document")
            parties = doc["metadata"].get("parties", "Unknown Parties")
            content = doc["content"]
            
            context_string += f"--- Document [{idx+1}] ---\n"
            context_string += f"Title: {title}\nParties: {parties}\n"
            context_string += f"Content: {content}\n\n"
            
            sources.append({
                "id": doc["id"],
                "title": title,
                "content": content,
                "score": doc.get("rerank_score", 0.0)
            })
            
        prompt = f"""You are an expert legal assistant. Your task is to answer the user's question using ONLY the provided context documents.
If the answer is not contained in the context, clearly state that you do not have enough information. Do not guess or make up legal precedents.

When answering:
1. Be precise and professional.
2. Cite the specific Document [number] and Title when quoting or referencing a specific clause.

Context Documents:
{context_string}"""

        if chat_history:
            history_str = "\n".join([f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in chat_history])
            prompt += f"\n\nPrevious Conversation:\n{history_str}\n"

        prompt += f"\nUser Question: {query}"

        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=500,
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            return {
                "answer": content,
                "sources": sources
            }
        except Exception as e:
            print(f"Generation failed: {e}")
            return {"answer": f"An error occurred during generation: {e}", "sources": sources}
