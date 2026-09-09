"""FastAPI boundary for the CaseGraph workflow."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .graph import GameState, GraphManager

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT / "static"


class NewGameRequest(BaseModel):
    num_suspects: int = Field(default=4, ge=3, le=6)


class AskRequest(BaseModel):
    game_id: str = Field(min_length=1, max_length=80)
    suspect_id: str = Field(pattern=r"^s\d+$")
    question: str = Field(min_length=2, max_length=500)


class AccuseRequest(BaseModel):
    game_id: str = Field(min_length=1, max_length=80)
    suspect_id: str = Field(pattern=r"^s\d+$")


def _messages(state: GameState) -> list[dict[str, Any]]:
    return [
        {
            "role": getattr(message, "type", ""),
            "name": getattr(message, "name", None),
            "content": getattr(message, "content", ""),
        }
        for message in state.get("messages", [])
    ]


def _public_case(game_id: str, state: GameState) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "summary": state["summary"],
        "details": state["details"],
        "suspects": [
            {"id": item["id"], "name": item["name"], "occupation": item["occupation"]}
            for item in state["suspects"]
        ],
        "suspicion": state["suspicion"],
        "contradictions": state["contradictions"],
        "turn_count": state["turn_count"],
    }


def create_app(manager: GraphManager | None = None) -> FastAPI:
    owns_manager = manager is None
    graph_manager = manager or GraphManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        if owns_manager:
            graph_manager.close()

    application = FastAPI(
        title="CaseGraph",
        version="1.0.0",
        description="A durable LangGraph-powered detective game.",
        lifespan=lifespan,
    )
    application.state.graph_manager = graph_manager

    origins = [
        item.strip()
        for item in os.environ.get(
            "CASEGRAPH_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
        ).split(",")
        if item.strip()
    ]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @application.get("/api/health")
    def health():
        return {"status": "ok", "workflow": "casegraph"}

    @application.post("/api/new_game")
    def new_game(request: NewGameRequest):
        game_id, state = graph_manager.new_game(request.num_suspects)
        return _public_case(game_id, state)

    @application.post("/api/ask")
    def ask(request: AskRequest):
        try:
            state = graph_manager.ask(request.game_id, request.suspect_id, request.question)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Game not found") from error
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Suspect not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error).capitalize()) from error

        return {
            "answer": state.get("last_answer", ""),
            "analysis": state.get("last_analysis"),
            "suspicion": state["suspicion"],
            "contradictions": state["contradictions"],
            "turn_count": state["turn_count"],
            "game_over": state["game_over"],
            "result": state["result"],
            "messages": _messages(state),
        }

    @application.post("/api/accuse")
    def accuse(request: AccuseRequest):
        try:
            state = graph_manager.accuse(request.game_id, request.suspect_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Game not found") from error
        except ValueError as error:
            raise HTTPException(status_code=404, detail="Suspect not found") from error
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error).capitalize()) from error

        return {
            "game_over": state["game_over"],
            "result": state["result"],
            "reveal": state["reveal"],
            "messages": _messages(state),
        }

    return application


app = create_app()
