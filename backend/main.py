from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

from .graph import GraphManager, SessionStore

# Load environment variables (e.g., DASHSCOPE_API_KEY) from .env if present
load_dotenv()
# Also load from qwen.env if present (since .env is a virtualenv directory in this repo)
try:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "qwen.env", override=True)
except Exception:
    pass


app = FastAPI(title="Detector Game")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files (frontend)
ROOT = Path(__file__).resolve().parent.parent
static_dir = ROOT / "static"
# Mount static assets under /static so API routes remain accessible
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Serve the SPA index at root
@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


graph_manager = GraphManager()
sessions = SessionStore()


class NewGameRequest(BaseModel):
    num_suspects: Optional[int] = 4


class AskRequest(BaseModel):
    game_id: str
    suspect_id: str
    question: str


class AccuseRequest(BaseModel):
    game_id: str
    suspect_id: str


@app.post("/api/new_game")
def new_game(req: NewGameRequest):
    payload = graph_manager.new_game(num_suspects=req.num_suspects or 4)
    game_id = sessions.create(payload)
    state = payload["state"]
    # Return public-facing info
    return {
        "game_id": game_id,
        "summary": state["summary"],
        "details": state["details"],
        "suspects": [
            {"id": s["id"], "name": s["name"], "occupation": s["occupation"]}
            for s in state["suspects"]
        ],
        "suspicion": state["suspicion"],
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    if req.game_id not in sessions.sessions:
        raise HTTPException(status_code=404, detail="Game not found")
    state = sessions.get_state(req.game_id)
    if state.get("game_over"):
        raise HTTPException(status_code=400, detail="Game is over")
    out = graph_manager.ask(state, req.suspect_id, req.question)
    sessions.set_state(req.game_id, out)
    # Find the last AI message content for convenience
    last_ai = next((m for m in reversed(out["messages"]) if getattr(m, "type", "ai") == "ai"), None)
    answer = getattr(last_ai, "content", "") if last_ai else ""
    return {
        "answer": answer,
        "suspicion": out["suspicion"],
        "game_over": out["game_over"],
        "result": out["result"],
        "messages": [
            {"role": getattr(m, "type", ""), "name": getattr(m, "name", None), "content": getattr(m, "content", "")}
            for m in out["messages"]
        ],
    }


@app.post("/api/accuse")
def accuse(req: AccuseRequest):
    if req.game_id not in sessions.sessions:
        raise HTTPException(status_code=404, detail="Game not found")
    state = sessions.get_state(req.game_id)
    if state.get("game_over"):
        raise HTTPException(status_code=400, detail="Game is over")
    out = graph_manager.accuse(state, req.suspect_id)
    sessions.set_state(req.game_id, out)
    return {
        "game_over": out["game_over"],
        "result": out["result"],
        "messages": [
            {"role": getattr(m, "type", ""), "name": getattr(m, "name", None), "content": getattr(m, "content", "")}
            for m in out["messages"]
        ],
    }