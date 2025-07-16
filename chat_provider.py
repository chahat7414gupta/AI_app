import os
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from vector_store import VectorStore

load_dotenv()


# === Base Provider Interface ===
class BaseProvider:
    async def get_response(self, prompt: str) -> str:
        raise NotImplementedError


# === OpenAI Chat Provider ===
class ChatProvider(BaseProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = VectorStore()

    async def get_response(self, prompt: str) -> str:
        context = self.vector_store.search(prompt)
        full_prompt = f"Context:\n{context}\n\nUser Query:\n{prompt}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI Error] {e}"


# === Grok Chat Provider ===
class GrokProvider(BaseProvider):
    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY")
        self.vector_store = VectorStore()

    async def get_response(self, prompt: str) -> str:
        context = self.vector_store.search(prompt)
        full_prompt = f"Context:\n{context}\n\nUser Query:\n{prompt}"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "command-r",  # or command-r-plus
            "messages": [{"role": "user", "content": full_prompt}]
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Grok Error] {str(e)}"


# === Provider switch utility ===
def get_provider(use_grok: bool = False) -> BaseProvider:
    return GrokProvider() if use_grok else ChatProvider()
