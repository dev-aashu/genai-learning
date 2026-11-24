"""
Step 09: FastAPI backend.
POST /chat  -> streams the model's answer token by token (from local Ollama).
GET  /health -> quick check that the backend is running.
"""
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ollama import chat
from pydantic import BaseModel

app = FastAPI(title="Local AI Chat Backend")

SYSTEM_PROMPT = "You are a helpful, concise assistant. Reply in plain text or simple markdown."


class ChatRequest(BaseModel):
    messages: list[dict]          # full chat history: [{"role": "user", "content": "..."}, ...]
    model: str = "llama3.2:3b"
    think: bool = False           # True only for thinking models like qwen3


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + req.messages

    def generate():
        try:
            # With think=True, Ollama puts the model's reasoning in chunk.message.thinking
            # and only the final answer in chunk.message.content, so we stream content only.
            for chunk in chat(model=req.model, messages=messages, stream=True, think=req.think):
                if chunk.message.content:
                    yield chunk.message.content
        except Exception as e:
            yield f"\n\n❌ Error from model: {e}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")