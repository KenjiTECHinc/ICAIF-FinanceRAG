from service.llm.gemini import GeminiClient

class ContextAwareRanker:
    def __init__(self):
        self.voter_1 = GeminiClient(model_name="gemini-2.5-flash", temperature=0.0, system_prompt="You are an expert ranker.")
    
    def rank(self, query: str, contexts: list[str]) -> list[str]:
        ranked_contexts = []
        for context in contexts:
            prompt = f"Given the query: '{query}', rank the following context: '{context}' based on relevance."
            score = self.voter_1.generate_text(prompt)
            ranked_contexts.append((context, score))
        
        # Sort contexts based on the generated scores (assuming higher score means more relevant)
        ranked_contexts.sort(key=lambda x: x[1], reverse=True)
        
        return [context for context, score in ranked_contexts]