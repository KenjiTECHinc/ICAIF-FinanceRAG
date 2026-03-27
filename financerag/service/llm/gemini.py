import os
from google import generativeai

class GeminiClient:
    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0, system_prompt: str = ""):
        self.client = self._initialize_client()
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt
        
        try:
            self.model = self._initialize_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {e}")
        

    def _initialize_client(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        client = generativeai.configure(api_key=api_key)
        return client
    
    def _initialize_model(self):
        model = generativeai.GenerativeModel(
            model_name=self.model_name, 
            system_instruction=self.system_prompt
            )
        return model
        

    def generate_text(self, prompt: str) -> str:
        response = generativeai.generate_content(
            prompt=prompt,
            generation_config={
                "temperature": self.temperature
            }
        )
        return response.text