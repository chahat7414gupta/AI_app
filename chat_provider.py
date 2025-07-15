import os
import httpx
from openai import OpenAI
from vector_store import VectorStore

vector_store = VectorStore()

# === Base Provider Interface ===
class ChatProvider:
    async def get_response(self, prompt: str) -> str:
        raise NotImplementedError

# === OpenAI Provider ===
class OpenAIProvider(ChatProvider):
    async def get_response(self, prompt: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in .env")

        try:
            context = vector_store.search(prompt)
            full_prompt = f"Context:\n{context}\n\nUser Query:\n{prompt}"

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": full_prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[OpenAI Error]: {str(e)}"

# === Grok Provider ===
class GrokProvider(ChatProvider):
    async def get_response(self, prompt: str) -> str:
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            raise ValueError("GROK_API_KEY is not set in .env")

        try:
            context = vector_store.search(prompt)
            full_prompt = f"Context:\n{context}\n\nUser Query:\n{prompt}"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": "command-r",
                "messages": [{"role": "user", "content": full_prompt}]
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=data
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            return f"[Grok Error]: {e.response.status_code} - {e.response.text}"
        except Exception as e:
            return f"[Grok Error]: {str(e)}"

# === Provider Selector ===
def get_provider(use_grok: bool) -> ChatProvider:
    return GrokProvider() if use_grok else OpenAIProvider()
