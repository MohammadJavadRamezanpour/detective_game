from typing import Any, Dict, List
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage


def generate_scenario(llm, num_suspects: int = 4) -> Dict[str, Any]:
    """Generate a deterministic scenario using the LLM.
    the LLM is invoked with temperature=0 for planning to
    keep outputs stable across runs.
    """
    # Prefer deterministic generation by binding temperature to 0
    try:
        deterministic = getattr(llm, "bind", lambda **kwargs: llm)(temperature=0.0)
    except Exception:
        deterministic = llm

    sys = SystemMessage(
        content=(
            "You are generating a grounded detective interrogation case for a web game. "
            "Output STRICT JSON ONLY (no markdown, no commentary). Schema:\n"
            "{\n  \"summary\": string,\n  \"details\": {\n    \"crime\": string,\n    \"location\": string,\n    \"time_window\": string,\n    \"clues\": [string, ...]\n  },\n  \"suspects\": [\n    { \"id\": 's1', \"name\": string, \"occupation\": string, \"bio\": string, \"alibi\": string, \"role\": 'suspect'|'criminal' }\n  ],\n  \"criminal_id\": 'sX'\n}\n"
            "Choose one suspect as the criminal. Keep facts consistent and plausible."
        )
    )
    hm = HumanMessage(
        content=(
            f"Create a case with {num_suspects} suspects. "
            "Avoid randomness; use consistent narrative and realistic names/occupations. "
            "Keep bios 1-2 sentences, alibis 1 sentence. Clues should be concrete and checkable."
        )
    )

    resp = deterministic.invoke([sys, hm])
    raw = getattr(resp, "content", "") or str(resp)
    # Strip code fences if present
    raw = re.sub(r"^```(json)?\n|\n```$", "", raw.strip())
    data = json.loads(raw)

    suspects: List[Dict[str, Any]] = data.get("suspects", [])
    if not suspects or len(suspects) != num_suspects:
        raise ValueError("invalid suspects count")
    for i, s in enumerate(suspects, start=1):
        s["id"] = f"s{i}"
        s.setdefault("role", "suspect")
    criminal_id = data.get("criminal_id") or next((s["id"] for s in suspects if s.get("role") == "criminal"), "s1")
    if criminal_id not in {s["id"] for s in suspects}:
        criminal_id = "s1"

    return {
        "summary": data.get("summary", ""),
        "details": data.get("details", {}),
        "suspects": suspects,
        "criminal_id": criminal_id,
    }