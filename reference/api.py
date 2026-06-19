"""
api.py — Layer 3: an OpenAI-compatible facade over the swarm.

Clients speak the chat-completions schema they already use; the daemon hides the
entire P2P pipeline behind it. Kept dependency-light: a plain callable plus an
optional FastAPI app (only imported if FastAPI is installed).
"""
import time
import uuid


def chat_completion(engine, model: str, messages: list[dict],
                    max_tokens: int = 64, temperature: float = 0.0,
                    client_id: str = "client") -> dict:
    prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    text, ids, stats = engine.generate(
        prompt, max_tokens=max_tokens, client_id=client_id,
        temperature=temperature)
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "length",
        }],
        "usage": {
            "completion_tokens": stats.tokens,
            "prompt_tokens": len(prompt),
            "total_tokens": stats.tokens + len(prompt),
        },
        "hopper": {                      # swarm telemetry, non-standard extension
            "audits": stats.audits,
            "audit_fails": stats.audit_fails,
            "reroutes": stats.reroutes,
            "network": stats.network,
        },
    }


def build_app(engine, model_name: str):
    """Return a FastAPI app exposing POST /v1/chat/completions, if FastAPI is
    available. Mirrors the OpenAI route so existing SDKs point straight at it."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    class Msg(BaseModel):
        role: str
        content: str

    class Req(BaseModel):
        model: str = model_name
        messages: list[Msg]
        max_tokens: int = 64
        temperature: float = 0.0

    app = FastAPI(title="HOPPER")

    @app.post("/v1/chat/completions")
    def completions(req: Req):
        return chat_completion(
            engine, req.model, [m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens, temperature=req.temperature)

    return app