from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from .llm_strategy import LLMStrategyFactory


class GameState(TypedDict):
    messages: List[Any]
    summary: str
    details: Dict[str, Any]
    suspects: List[Dict[str, Any]]
    criminal_id: str
    suspicion: Dict[str, float]
    latest_user_question: Optional[str]
    target: Optional[str]
    last_answer: Optional[str]
    accused: Optional[str]
    game_over: bool
    result: Optional[str]


class GraphManager:
    def __init__(self) -> None:
        self.llm_strategy = LLMStrategyFactory.create_strategy()
        # The graph is stateless; we rebuild per invoke via compiled self.graph
        self.graph = self._build_graph()

    def _build_graph(self):
        sg = StateGraph(GameState)

        def user_input_node(state: GameState) -> Dict[str, Any]:
            # Pass through; routing handled by edges
            return {}

        def suspect_answer_node(state: GameState) -> Dict[str, Any]:
            target_id = state.get("target")
            question = state.get("latest_user_question") or ""
            suspects = state.get("suspects", [])
            scenario = {
                "summary": state.get("summary", ""),
                "details": state.get("details", {}),
            }
            suspect = next((s for s in suspects if s["id"] == target_id), None)
            if not suspect:
                # No target; do nothing
                return {}

            answer = self.llm_strategy.suspect_reply(suspect, scenario, question, state.get("messages", []))
            # Append AI message with name metadata
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=answer, name=suspect.get("name", "Suspect")))
            return {"messages": messages, "last_answer": answer}

        def update_suspicion_node(state: GameState) -> Dict[str, Any]:
            target_id = state.get("target")
            suspects = state.get("suspects", [])
            suspect = next((s for s in suspects if s["id"] == target_id), None)
            if not suspect:
                return {}
            current = float(state.get("suspicion", {}).get(target_id, 0.0))
            delta = self.llm_strategy.analyze_suspicion(
                {"summary": state.get("summary", "")},
                suspect,
                state.get("last_answer", ""),
                state.get("latest_user_question", ""),
                current,
            )
            new_val = max(0.0, min(current + delta, 10.0))
            suspicion = dict(state.get("suspicion", {}))
            suspicion[target_id] = new_val
            return {"suspicion": suspicion}

        def accuse_check_node(state: GameState) -> Dict[str, Any]:
            accused = state.get("accused")
            if not accused:
                return {}
            criminal_id = state.get("criminal_id")
            messages = list(state.get("messages", []))
            if accused == criminal_id:
                reveal = (
                    "Correct! You identified the criminal. The case is closed."
                )
                result = "win"
            else:
                reveal = (
                    "Incorrect accusation. The real criminal slips away for now."
                )
                result = "lose"
            messages.append(AIMessage(content=reveal, name="Case"))
            return {"messages": messages, "game_over": True, "result": result}

        def router(state: GameState) -> str:
            # If the user accused someone, skip Q&A and go to accuse check
            if state.get("accused"):
                return "accuse_check"
            # Otherwise answer and update suspicion
            return "suspect_answer"

        sg.add_node("user_input", user_input_node)
        sg.add_node("suspect_answer", suspect_answer_node)
        sg.add_node("update_suspicion", update_suspicion_node)
        sg.add_node("accuse_check", accuse_check_node)

        sg.set_entry_point("user_input")
        sg.add_conditional_edges("user_input", router, {"suspect_answer": "suspect_answer", "accuse_check": "accuse_check"})
        sg.add_edge("suspect_answer", "update_suspicion")
        sg.add_edge("update_suspicion", END)
        sg.add_edge("accuse_check", END)

        return sg.compile()

    def new_game(self, num_suspects: int = 4) -> Dict[str, Any]:
        scenario = self.llm_strategy.generate_scenario(num_suspects=num_suspects)
        suspicion = {s["id"]: 0.0 for s in scenario["suspects"]}
        state: GameState = {
            "messages": [],
            "summary": scenario["summary"],
            "details": scenario["details"],
            "suspects": scenario["suspects"],
            "criminal_id": scenario["criminal_id"],
            "suspicion": suspicion,
            "latest_user_question": None,
            "target": None,
            "last_answer": None,
            "accused": None,
            "game_over": False,
            "result": None,
        }
        game_id = str(uuid.uuid4())
        return {"game_id": game_id, "state": state}

    def ask(self, state: GameState, suspect_id: str, question: str) -> GameState:
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=question, name="Player"))
        state.update({
            "messages": messages,
            "latest_user_question": question,
            "target": suspect_id,
        })
        out = self.graph.invoke(state)
        # Clear transient fields
        out["latest_user_question"] = None
        out["target"] = None
        return out

    def accuse(self, state: GameState, suspect_id: str) -> GameState:
        state.update({"accused": suspect_id})
        out = self.graph.invoke(state)
        return out


class SessionStore:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def create(self, payload: Dict[str, Any]):
        game_id = payload["game_id"]
        self.sessions[game_id] = payload
        return game_id

    def get_state(self, game_id: str) -> GameState:
        return self.sessions[game_id]["state"]

    def set_state(self, game_id: str, state: GameState):
        self.sessions[game_id]["state"] = state