import os
from huggingface_hub import InferenceClient

class QueryRouter:
    def __init__(self):
        self.api_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if not self.api_token:
            print("WARNING: HUGGINGFACEHUB_API_TOKEN not found. Router will default to 'balanced'.")
            self.client = None
        else:
            print("Initializing HuggingFace Router (Qwen)...")
            self.client = InferenceClient(token=self.api_token)
            self.model = "Qwen/Qwen2.5-72B-Instruct"
            
    def route_query(self, query: str) -> str:
        if not self.client:
            return "balanced"
            
        prompt = f"""You are a routing assistant for a legal document search engine.
Classify the following user query into EXACTLY ONE of these three categories:
1. 'keyword_heavy': The user is looking for a specific section number, exact clause, name, or explicit phrase (e.g., "Find Section 4.2", "Acme Corp NDA").
2. 'semantic_heavy': The user is asking a conceptual question or describing a scenario without knowing the exact legal terms (e.g., "What happens if I quit early?").
3. 'balanced': The query contains a mix of both, or is a general inquiry.

Output ONLY the exact string 'keyword_heavy', 'semantic_heavy', or 'balanced'.

Query: "{query}" """
        try:
            response = self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                max_tokens=10,
                temperature=0.01
            )
            content = response.choices[0].message.content.strip().lower()
            print(f"Raw Router Response: {content}")
            
            if "keyword_heavy" in content: return "keyword_heavy"
            if "semantic_heavy" in content: return "semantic_heavy"
            return "balanced"
        except Exception as e:
            print(f"Routing failed: {e}. Defaulting to 'balanced'.")
            return "balanced"
