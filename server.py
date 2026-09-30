"""One small service: serves the static web app and proxies the LLM call (keeps your API key off the client).
Run locally:  uvicorn server:app --port 8000   ->  http://localhost:8000"""
from pathlib import Path
from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import llm

app = FastAPI()


class Req(BaseModel):
    tokens: List[List[Tuple[str, float]]]      # per sign: [[gloss, confidence], ...]


@app.post("/api/translate")
def translate(req: Req):
    if not (0 < len(req.tokens) <= 20) or any(len(c) == 0 or len(c) > 5 for c in req.tokens):
        raise HTTPException(400, "bad tokens")
    toks = [[(g[:40], float(c)) for g, c in cands] for cands in req.tokens]
    return llm.gloss_to_sentence(toks)


@app.get("/api/health")
def health():
    return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")
