"""Durable LangGraph workflow for creating and investigating cases."""

from __future__ import annotations

import operator
import os
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .llm_strategy import BaseLLMStrategy, LLMStrategyFactory


class GameState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    analysis_log: Annotated[list[dict[str, Any]], operator.add]
    action: Literal["new_game", "question", "accuse"]
    requested_suspects: int
    summary: str
    details: dict[str, Any]
    suspects: list[dict[str, Any]]
    criminal_id: str
    suspicion: dict[str, float]
    suspect_histories: dict[str, list[dict[str, str]]]
    contradictions: list[dict[str, str]]
    latest_user_question: str | None
    target: str | None
    last_answer: str | None
    last_analysis: dict[str, Any] | None
    accused: str | None
    game_over: bool
    result: Literal["win", "lose"] | None
    reveal: dict[str, Any] | None
    turn_count: int


class GraphManager:
    """Own the compiled graph and address games through checkpoint thread IDs."""

    def __init__(
        self,
        llm_strategy: BaseLLMStrategy | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        self.llm_strategy = llm_strategy or LLMStrategyFactory.create_strategy()
        self._connection: sqlite3.Connection | None = None

        if checkpointer is None:
            configured_path = db_path or os.environ.get(
                "CASEGRAPH_DB_PATH", ".data/casegraph.sqlite"
            )
            checkpoint_path = Path(configured_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            checkpointer = SqliteSaver(self._connection)

        self.checkpointer = checkpointer
        self.graph = self._build_graph().compile(checkpointer=self.checkpointer)

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(GameState)

        def create_case(state: GameState) -> dict[str, Any]:
            scenario = self.llm_strategy.generate_scenario(state.get("requested_suspects", 4))
            data = scenario.model_dump()
            return {
                **data,
                "messages": [],
                "analysis_log": [],
                "suspicion": {suspect["id"]: 0.0 for suspect in data["suspects"]},
                "suspect_histories": {suspect["id"]: [] for suspect in data["suspects"]},
                "contradictions": [],
                "latest_user_question": None,
                "target": None,
                "last_answer": None,
                "last_analysis": None,
                "accused": None,
                "game_over": False,
                "result": None,
                "reveal": None,
                "turn_count": 0,
            }

        def answer_suspect(state: GameState) -> dict[str, Any]:
            suspect = self._find_suspect(state, state.get("target"))
            question = state.get("latest_user_question") or ""
            histories = {key: list(value) for key, value in state["suspect_histories"].items()}
            suspect_history = histories[suspect["id"]]
            scenario = {"summary": state["summary"], "details": state["details"]}
            answer = self.llm_strategy.suspect_reply(suspect, scenario, question, suspect_history)
            histories[suspect["id"]] = [
                *suspect_history,
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ]
            return {
                "messages": [AIMessage(content=answer, name=suspect["name"])],
                "suspect_histories": histories,
                "last_answer": answer,
                "turn_count": state.get("turn_count", 0) + 1,
            }

        def analyze_evidence(state: GameState) -> dict[str, Any]:
            suspect = self._find_suspect(state, state.get("target"))
            target_id = suspect["id"]
            scenario = {"summary": state["summary"], "details": state["details"]}
            current = float(state["suspicion"].get(target_id, 0.0))
            analysis = self.llm_strategy.analyze_evidence(
                scenario,
                suspect,
                state.get("last_answer") or "",
                state.get("latest_user_question") or "",
                state["suspect_histories"][target_id],
                current,
            )
            analysis_data = analysis.model_dump()

            valid_clues = set(state["details"].get("clues", []))
            if analysis_data.get("relevant_clue") not in valid_clues:
                analysis_data["relevant_clue"] = None

            suspicion = dict(state["suspicion"])
            suspicion[target_id] = round(max(0.0, min(10.0, current + analysis.suspicion_delta)), 2)

            event = {
                "suspect_id": target_id,
                "suspect_name": suspect["name"],
                **analysis_data,
                "score": suspicion[target_id],
            }
            contradictions = list(state.get("contradictions", []))
            contradiction = analysis_data.get("contradiction")
            if contradiction and not any(item["text"] == contradiction for item in contradictions):
                contradictions.append(
                    {
                        "suspect_id": target_id,
                        "suspect_name": suspect["name"],
                        "text": contradiction,
                    }
                )

            return {
                "suspicion": suspicion,
                "last_analysis": event,
                "analysis_log": [event],
                "contradictions": contradictions,
                "latest_user_question": None,
                "target": None,
            }

        def check_accusation(state: GameState) -> dict[str, Any]:
            accused = self._find_suspect(state, state.get("accused"))
            criminal = self._find_suspect(state, state["criminal_id"])
            won = accused["id"] == criminal["id"]
            result: Literal["win", "lose"] = "win" if won else "lose"
            verdict = (
                f"Correct. {criminal['name']} committed the crime. Case closed."
                if won
                else f"That accusation was incorrect. The evidence points to {criminal['name']}."
            )
            reveal = {
                "criminal": criminal["name"],
                "criminal_id": criminal["id"],
                "alibi": criminal["alibi"],
                "clues": state["details"].get("clues", []),
            }
            return {
                "messages": [AIMessage(content=verdict, name="Case")],
                "game_over": True,
                "result": result,
                "reveal": reveal,
            }

        def route_action(state: GameState) -> str:
            action = state.get("action")
            if action not in {"new_game", "question", "accuse"}:
                raise ValueError(f"unsupported graph action: {action}")
            return action

        workflow.add_node("create_case", create_case)
        workflow.add_node("answer_suspect", answer_suspect)
        workflow.add_node("analyze_evidence", analyze_evidence)
        workflow.add_node("check_accusation", check_accusation)
        workflow.add_conditional_edges(
            START,
            route_action,
            {"new_game": "create_case", "question": "answer_suspect", "accuse": "check_accusation"},
        )
        workflow.add_edge("create_case", END)
        workflow.add_edge("answer_suspect", "analyze_evidence")
        workflow.add_edge("analyze_evidence", END)
        workflow.add_edge("check_accusation", END)
        return workflow

    @staticmethod
    def _find_suspect(state: GameState, suspect_id: str | None) -> dict[str, Any]:
        suspect = next(
            (item for item in state.get("suspects", []) if item["id"] == suspect_id), None
        )
        if suspect is None:
            raise ValueError("unknown suspect")
        return suspect

    @staticmethod
    def _config(game_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": game_id}}

    def new_game(self, num_suspects: int = 4) -> tuple[str, GameState]:
        if not 3 <= num_suspects <= 6:
            raise ValueError("num_suspects must be between 3 and 6")
        game_id = str(uuid.uuid4())
        state = self.graph.invoke(
            {"action": "new_game", "requested_suspects": num_suspects}, self._config(game_id)
        )
        return game_id, state

    def get_state(self, game_id: str) -> GameState:
        values = self.graph.get_state(self._config(game_id)).values
        if not values:
            raise KeyError(game_id)
        return values

    def ask(self, game_id: str, suspect_id: str, question: str) -> GameState:
        current = self.get_state(game_id)
        if current.get("game_over"):
            raise RuntimeError("game is over")
        self._find_suspect(current, suspect_id)
        return self.graph.invoke(
            {
                "action": "question",
                "target": suspect_id,
                "latest_user_question": question,
                "messages": [HumanMessage(content=question, name="Player")],
            },
            self._config(game_id),
        )

    def accuse(self, game_id: str, suspect_id: str) -> GameState:
        current = self.get_state(game_id)
        if current.get("game_over"):
            raise RuntimeError("game is over")
        self._find_suspect(current, suspect_id)
        return self.graph.invoke(
            {"action": "accuse", "accused": suspect_id}, self._config(game_id)
        )

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
