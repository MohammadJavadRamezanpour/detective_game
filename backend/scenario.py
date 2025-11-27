from typing import Any, Dict, List
import json
import re

from langchain_core.messages import HumanMessage, SystemMessage


CRIMES = [
    "A rare painting stolen from a private gallery",
    "A high-end laptop missing from a tech startup's lab",
    "A diamond necklace stolen during a charity gala",
    "Confidential documents leaked from a law firm",
    "An antique watch missing from a family mansion",
]

PLACES = [
    "Riverside Mansion",
    "Old Town Gallery",
    "Harbor Conference Center",
    "City Loft Co-working Space",
    "Grand Oak Estate",
]

OCCUPATIONS = [
    "event coordinator",
    "security guard",
    "software engineer",
    "art curator",
    "journalist",
    "caterer",
    "photographer",
    "law student",
]


def _random_names(n: int) -> List[str]:
    first = [
        "Ava",
        "Liam",
        "Noah",
        "Mia",
        "Ethan",
        "Zoe",
        "Leo",
        "Nora",
        "Ivy",
        "Kai",
    ]
    last = [
        "Morgan",
        "Reed",
        "Patel",
        "Kim",
        "Lopez",
        "Baker",
        "Shaw",
        "Nguyen",
        "Carter",
        "Ali",
    ]
    random.shuffle(first)
    random.shuffle(last)
    return [f"{first[i % len(first)]} {last[i % len(last)]}" for i in range(n)]


def _llm_summary(llm, crime: str, place: str) -> str:
    try:
        sys = SystemMessage(
            content=(
                "You are a crime storyteller. Write a 3-4 sentence summary of the case. "
                "Keep it grounded and concise."
            )
        )
        hm = HumanMessage(
            content=f"Crime: {crime}. Location: {place}. Include time window and notable clues."
        )
        resp = llm.invoke([sys, hm])
        return getattr(resp, "content", "")
    except Exception:
        return (
            f"{crime} at {place}. The incident occurred in the early evening. "
            "Witnesses reported hurried footsteps and a vehicle leaving the area. "
            "Several items were found out of place, and access logs show unusual activity."
        )


def _llm_bio(llm, name: str, occupation: str) -> str:
    try:
        sys = SystemMessage(
            content=(
                "You are creating short suspect bios. Return 1-2 sentences with a distinct trait."
            )
        )
        hm = HumanMessage(content=f"Name: {name}, Occupation: {occupation}")
        resp = llm.invoke([sys, hm])
        return getattr(resp, "content", f"{name} is a {occupation} with a quiet demeanor.")
    except Exception:
        return f"{name} is a {occupation} known for being meticulous and private."


def generate_scenario(llm, num_suspects: int = 4) -> Dict[str, Any]:
    """Generate a deterministic scenario using the LLM (Qwen preferred).

    No use of Python's random; the LLM is invoked with temperature=0 for planning to
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

    # try:
    resp = deterministic.invoke([sys, hm])
    raw = getattr(resp, "content", "") or str(resp)
    # Strip code fences if present
    raw = re.sub(r"^```(json)?\n|\n```$", "", raw.strip())
    data = json.loads(raw)

    # Basic validation / normalization
    suspects: List[Dict[str, Any]] = data.get("suspects", [])
    if not suspects or len(suspects) != num_suspects:
        raise ValueError("invalid suspects count")
    # Ensure ids are s1..sN
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
    # except Exception:
    #     # Deterministic fallback (no randomness): fixed scenario
    #     suspect_list = [
    #         {
    #             "id": "s1",
    #             "name": "Alex Reed",
    #             "occupation": "event coordinator",
    #             "bio": "Alex manages logistics and tends to stay calm under pressure.",
    #             "alibi": "Was checking the supply inventory near the service corridor.",
    #             "role": "suspect",
    #         },
    #         {
    #             "id": "s2",
    #             "name": "Maya Kim",
    #             "occupation": "art curator",
    #             "bio": "Maya oversees exhibits and is meticulous about security protocols.",
    #             "alibi": "Stepped out to call a vendor and returned just before the incident.",
    #             "role": "criminal",
    #         },
    #         {
    #             "id": "s3",
    #             "name": "Jordan Lopez",
    #             "occupation": "photographer",
    #             "bio": "Jordan documents events and moves between rooms frequently.",
    #             "alibi": "Was setting up lighting near the main hall.",
    #             "role": "suspect",
    #         },
    #         {
    #             "id": "s4",
    #             "name": "Sam Patel",
    #             "occupation": "security guard",
    #             "bio": "Sam patrols entrances and monitors access logs.",
    #             "alibi": "Circulating between checkpoints and scanning badges.",
    #             "role": "suspect",
    #         },
    #     ][:num_suspects]

    #     details = {
    #         "crime": "A rare painting stolen from a private gallery",
    #         "location": "Old Town Gallery",
    #         "time_window": "Between 7:30pm and 8:15pm",
    #         "clues": [
    #             "Access badge used near the gallery at 7:52pm",
    #             "Footprints near the side exit",
    #             "A receipt found in the lounge",
    #         ],
    #     }
    #     summary = (
    #         "During a crowded exhibition at the Old Town Gallery, a prized painting vanished. "
    #         "Staff reports indicate unusual badge activity and a brief power fluctuation. "
    #         "Several attendees and workers had plausible reasons to move through restricted areas."
    #     )
    #     criminal_id = next((s["id"] for s in suspect_list if s.get("role") == "criminal"), suspect_list[0]["id"])
    #     return {
    #         "summary": summary,
    #         "details": details,
    #         "suspects": suspect_list,
    #         "criminal_id": criminal_id,
    #     }