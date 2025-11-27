import os
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


def _get_qwen_key() -> str:
    """Return Qwen/DashScope API key if present.

    Supports environment variables:
    - `DASHSCOPE_API_KEY` (official)
    - `QWEN_API_KEY` (alias)
    """
    return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or ""


def _has_openai_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _get_google_key() -> str:
    """Return Google API key if present."""
    return os.environ.get("GOOGLE_API_KEY") or ""


class MockLLM:
    """A minimal mock LLM used when no API key is provided."""

    def invoke(self, messages: List[Any]) -> Any:
        # Very simple persona-aware response using the last HumanMessage
        system = next((m for m in messages if isinstance(m, SystemMessage)), None)
        human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        persona_hint = system.content if system else ""
        question = human.content if human else ""
        content = (
            f"[Mock Response] {persona_hint[:120]}\n"
            f"I hear your question: '{question}'. "
            f"I don't recall anything suspicious. I was busy with my own tasks."
        )
        class _Resp:
            def __init__(self, c):
                self.content = c
        return _Resp(content)


def get_llm():
    """Return a ChatOpenAI-compatible object with .invoke(messages).

    Preference order:
    1) Qwen via OpenAI-compatible endpoint if `DASHSCOPE_API_KEY`/`QWEN_API_KEY` set
    2) OpenAI if `OPENAI_API_KEY` set
    3) MockLLM otherwise
    """
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return MockLLM()

    # Prefer Google Gemini if key is available
    google_key = _get_google_key()
    if google_key and ChatGoogleGenerativeAI:
        print("============= google_key found", google_key)
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            google_api_key=google_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )

    # Prefer Qwen if a Qwen/DashScope key is available
    qwen_key = _get_qwen_key()
    
    if qwen_key:
        base_url = (
            os.environ.get("QWEN_BASE_URL")
            or os.environ.get("DASHSCOPE_BASE_URL")
            or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        model = os.environ.get("QWEN_MODEL", "qwen-plus")
        try:
            return ChatOpenAI(api_key=qwen_key, base_url=base_url, model=model, temperature=0.7)
        except Exception:
            return MockLLM()

    # Otherwise fall back to OpenAI if available
    if _has_openai_key():
        try:
            return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        except Exception:
            return MockLLM()

    # No keys found
    return MockLLM()


def suspect_reply(
    llm: Any,
    suspect: Dict[str, Any],
    scenario: Dict[str, Any],
    question: str,
    chat_history: List[Any],
) -> str:
    """Generate a suspect's reply guided by persona and scenario."""
    role = suspect.get("role", "suspect")
    persona = suspect.get("bio", "")
    alibi = suspect.get("alibi", "")
    name = suspect.get("name", "Suspect")
    crime_summary = scenario.get("summary", "")
    crime_details = scenario.get("details", {})

    system_prompt = (
        "You are role-playing as a suspect in an interrogation game. "
        "Stay in character, use first person, and defend yourself. "
        "Do NOT confess unless the evidence is overwhelming and directly proves guilt. "
        "Keep responses concise (2-5 sentences). "
        f"Your name is {name}. Persona: {persona}. Alibi: {alibi}. "
        f"Case: {crime_summary}. Relevant facts: {crime_details}. "
        + (
            "As the criminal, be evasive, plausible, and deflect; avoid obvious contradictions. "
            if role == "criminal"
            else "As an innocent suspect, be cooperative and consistent. "
        )
    )

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=question)]
    resp = llm.invoke(messages)
    return getattr(resp, "content", str(resp))


def analyze_suspicion(
    llm: Any,
    scenario: Dict[str, Any],
    suspect: Dict[str, Any],
    last_answer: str,
    last_question: str,
    current_score: float,
) -> float:
    """Return a small delta to adjust suspicion score based on the latest exchange.

    If using the mock LLM, apply a lightweight heuristic.
    """
    # Heuristic for mock
    if isinstance(llm, MockLLM):
        delta = 0.0
        lower = (last_answer or "").lower()
        # crude signals
        suspicious_markers = ["avoid", "confuse", "contrad", "maybe", "think", "unsure", "don't recall", "forgot"]
        for w in suspicious_markers:
            if w in lower:
                delta += 0.3
        if "alibi" in lower and ("changed" in lower or "different" in lower):
            delta += 0.5
        return min(max(delta, -0.2), 0.8)

    # Ask LLM for a numeric delta [-1.0, 1.0] but keep small
    sys = SystemMessage(
        content=(
            "You are scoring interrogation answers for suspicion. "
            "Return ONLY a single float delta in the range [-0.5, 0.8] based on whether the answer increases suspicion. "
            "Small values are preferred; 0.0 if neutral."
        )
    )
    hm = HumanMessage(
        content=(
            f"Question: {last_question}\nAnswer: {last_answer}\n"
            f"Suspect persona: {suspect.get('bio','')}\n"
            f"Scenario summary: {scenario.get('summary','')}\n"
            f"Current suspicion: {current_score:.2f}"
        )
    )
    try:
        resp = llm.invoke([sys, hm])
        text = getattr(resp, "content", "0.0").strip()
        # Extract the first numeric value
        import re
        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
        if m:
            val = float(m.group(0))
            return float(max(min(val, 0.8), -0.5))
    except Exception:
        pass
    return 0.0