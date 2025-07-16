from fastapi import FastAPI, Request
from chat_provider import get_provider

app = FastAPI()
provider = get_provider(use_grok=False)  # 👈 switch this to `True` to use Grok

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    if not prompt:
        return {"error": "Prompt is required"}

    response = await provider.get_response(prompt)
    return {"response": response}
