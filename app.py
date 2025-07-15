from fastapi import FastAPI, Query
from pydantic import BaseModel

from chat_provider import get_provider

app = FastAPI()

class ChatRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: ChatRequest, use_grok: bool = Query(False)):
    # Just forward the user's prompt to the selected provider
    provider = get_provider(use_grok)
    reply = await provider.get_response(request.prompt)

    return {
        "reply": reply,
        "provider": "grok" if use_grok else "openai"
    }
