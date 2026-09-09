from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from backend.api import create_app
from backend.graph import GraphManager
from backend.llm_strategy import MockLLMStrategy


def build_client():
    manager = GraphManager(MockLLMStrategy(), InMemorySaver())
    return TestClient(create_app(manager))


def test_complete_api_flow_hides_answer_until_accusation():
    with build_client() as client:
        created = client.post("/api/new_game", json={"num_suspects": 4})
        assert created.status_code == 200
        case = created.json()
        assert "criminal_id" not in case
        assert all("role" not in suspect for suspect in case["suspects"])
        assert len(case["details"]["clues"]) == 3

        asked = client.post(
            "/api/ask",
            json={
                "game_id": case["game_id"],
                "suspect_id": "s2",
                "question": "When was your contractor badge used?",
            },
        )
        assert asked.status_code == 200
        assert asked.json()["analysis"]["contradiction_detected"] is True

        accused = client.post(
            "/api/accuse", json={"game_id": case["game_id"], "suspect_id": "s2"}
        )
        assert accused.status_code == 200
        assert accused.json()["result"] == "win"
        assert accused.json()["reveal"]["criminal_id"] == "s2"


def test_api_validates_inputs_and_missing_games():
    with build_client() as client:
        assert client.post("/api/new_game", json={"num_suspects": 12}).status_code == 422
        response = client.post(
            "/api/ask",
            json={"game_id": "missing", "suspect_id": "s1", "question": "Where were you?"},
        )
        assert response.status_code == 404
