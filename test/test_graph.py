from langgraph.checkpoint.memory import InMemorySaver

from backend.graph import GraphManager
from backend.llm_strategy import MockLLMStrategy


def build_manager(checkpointer=None):
    return GraphManager(MockLLMStrategy(), checkpointer or InMemorySaver())


def test_new_case_is_checkpointed_and_public_state_is_consistent():
    manager = build_manager()
    game_id, created = manager.new_game(4)

    restored = manager.get_state(game_id)
    assert restored["summary"] == created["summary"]
    assert restored["turn_count"] == 0
    assert restored["game_over"] is False
    assert set(restored["suspect_histories"]) == {"s1", "s2", "s3", "s4"}


def test_interrogations_accumulate_messages_and_isolate_suspect_memory():
    manager = build_manager()
    game_id, _ = manager.new_game(4)

    first = manager.ask(game_id, "s1", "When did you leave?")
    assert len(first["messages"]) == 2
    assert len(first["suspect_histories"]["s1"]) == 2
    assert first["suspect_histories"]["s2"] == []

    second = manager.ask(game_id, "s2", "Which badge did you use?")
    assert len(second["messages"]) == 4
    assert len(second["suspect_histories"]["s1"]) == 2
    assert len(second["suspect_histories"]["s2"]) == 2
    assert second["turn_count"] == 2
    assert second["last_analysis"]["contradiction_detected"] is True
    assert second["contradictions"][0]["suspect_id"] == "s2"


def test_accusation_routes_to_win_and_reveals_case():
    manager = build_manager()
    game_id, _ = manager.new_game(4)

    result = manager.accuse(game_id, "s2")
    assert result["game_over"] is True
    assert result["result"] == "win"
    assert result["reveal"]["criminal_id"] == "s2"


def test_sqlite_checkpoints_survive_manager_restart(tmp_path):
    database = tmp_path / "checkpoints.sqlite"
    first_manager = GraphManager(MockLLMStrategy(), db_path=database)
    game_id, created = first_manager.new_game(3)
    first_manager.close()

    second_manager = GraphManager(MockLLMStrategy(), db_path=database)
    restored = second_manager.get_state(game_id)
    second_manager.close()

    assert restored["summary"] == created["summary"]
    assert len(restored["suspects"]) == 3


def test_graph_exposes_portfolio_workflow_nodes():
    diagram = build_manager().graph.get_graph().draw_mermaid()
    assert "create_case" in diagram
    assert "answer_suspect" in diagram
    assert "analyze_evidence" in diagram
    assert "check_accusation" in diagram
