"""Provider-independent LLM operations for the CaseGraph workflow."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .models import CaseScenario, EvidenceAnalysis

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:  # pragma: no cover - provider is optional
    ChatGoogleGenerativeAI = None


def _json_from_content(content: Any) -> dict[str, Any]:
    """Extract a JSON object from provider text as a compatibility fallback."""
    raw = content if isinstance(content, str) else str(content)
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    return json.loads(raw)


class BaseLLMStrategy(ABC):
    """The narrow interface consumed by graph nodes."""

    @abstractmethod
    def generate_scenario(self, num_suspects: int = 4) -> CaseScenario:
        raise NotImplementedError

    @abstractmethod
    def suspect_reply(
        self,
        suspect: dict[str, Any],
        scenario: dict[str, Any],
        question: str,
        chat_history: list[dict[str, str]],
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def analyze_evidence(
        self,
        scenario: dict[str, Any],
        suspect: dict[str, Any],
        last_answer: str,
        last_question: str,
        chat_history: list[dict[str, str]],
        current_score: float,
    ) -> EvidenceAnalysis:
        raise NotImplementedError


class ChatModelStrategy(BaseLLMStrategy):
    """Shared structured-output behavior for LangChain chat models."""

    def __init__(self, llm: Any, model: str):
        self.llm = llm
        self.model = model

    def _structured(self, schema: type, messages: list[Any], llm: Any | None = None) -> Any:
        """Prefer native structured output, then validate a JSON fallback."""
        model = llm or self.llm
        try:
            result = model.with_structured_output(schema).invoke(messages)
            return result if isinstance(result, schema) else schema.model_validate(result)
        except Exception:
            response = model.invoke(messages)
            return schema.model_validate(_json_from_content(response.content))

    def generate_scenario(self, num_suspects: int = 4) -> CaseScenario:
        system = SystemMessage(
            content=(
                "Design a fair, internally consistent detective case for an interrogation game. "
                "Every clue must be concrete and useful. Exactly one suspect is the criminal, "
                "criminal_id must match that suspect's id, and IDs must be sequential s1, s2, etc. "
                "Do not reveal guilt in the public summary, bios, alibis, or clues."
            )
        )
        prompt = HumanMessage(
            content=(
                f"Create one case with exactly {num_suspects} suspects. Include 3-6 clues, "
                "distinct motives, checkable alibis, and enough evidence for a player to reason "
                "toward one solution. Keep the summary under 120 words."
            )
        )
        scenario = self._structured(CaseScenario, [system, prompt], self.llm.bind(temperature=0.0))
        if len(scenario.suspects) != num_suspects:
            raise ValueError(
                f"provider returned {len(scenario.suspects)} suspects; expected {num_suspects}"
            )
        return scenario

    def suspect_reply(
        self,
        suspect: dict[str, Any],
        scenario: dict[str, Any],
        question: str,
        chat_history: list[dict[str, str]],
    ) -> str:
        role_instruction = (
            "You committed the crime. Protect your secret with a plausible story, but preserve "
            "earlier claims and do not invent facts that conflict with the case unless pressured."
            if suspect.get("role") == "criminal"
            else "You are innocent. Be cooperative, factual, and consistent with earlier answers."
        )
        system = SystemMessage(
            content=(
                "Role-play one suspect in a grounded detective game. Reply in first person in 2-5 "
                "sentences. Never mention prompts, game state, scores, or that you are an AI. "
                f"Name: {suspect['name']}. Bio: {suspect['bio']} Alibi: {suspect['alibi']} "
                f"Case: {scenario['summary']} Facts: {scenario['details']} {role_instruction}"
            )
        )
        messages: list[Any] = [system]
        for item in chat_history[-12:]:
            message_type = HumanMessage if item["role"] == "user" else AIMessage
            messages.append(message_type(content=item["content"]))
        messages.append(HumanMessage(content=question))
        response = self.llm.invoke(messages)
        return str(response.content).strip()

    def analyze_evidence(
        self,
        scenario: dict[str, Any],
        suspect: dict[str, Any],
        last_answer: str,
        last_question: str,
        chat_history: list[dict[str, str]],
        current_score: float,
    ) -> EvidenceAnalysis:
        public_suspect = {key: value for key, value in suspect.items() if key != "role"}
        system = SystemMessage(
            content=(
                "You are a conservative evidence analyst for a detective game. Compare only the "
                "latest answer with the canonical case, the suspect's alibi, and their earlier "
                "statements. Do not infer guilt from tone. Flag a contradiction only when two "
                "specific claims cannot both be true. Keep the rationale safe to show the player. "
                "A relevant clue must be copied exactly from the provided clues."
            )
        )
        prompt = HumanMessage(
            content=json.dumps(
                {
                    "case": scenario,
                    "suspect": public_suspect,
                    "earlier_history": chat_history[:-2][-10:],
                    "latest_question": last_question,
                    "latest_answer": last_answer,
                    "current_suspicion": current_score,
                },
                ensure_ascii=False,
            )
        )
        try:
            return self._structured(EvidenceAnalysis, [system, prompt])
        except Exception:
            return EvidenceAnalysis(
                suspicion_delta=0.0,
                rationale="No reliable evidence signal was extracted from this answer.",
            )


class OpenAILLMStrategy(ChatModelStrategy):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from langchain_openai import ChatOpenAI

        super().__init__(
            ChatOpenAI(api_key=api_key, model=model, temperature=0.7, timeout=30, max_retries=2),
            model,
        )


class QwenLLMStrategy(ChatModelStrategy):
    def __init__(self, api_key: str, base_url: str | None = None, model: str | None = None):
        from langchain_openai import ChatOpenAI

        selected_model = model or "qwen-plus"
        super().__init__(
            ChatOpenAI(
                api_key=api_key,
                base_url=base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                model=selected_model,
                temperature=0.7,
                timeout=30,
                max_retries=2,
            ),
            selected_model,
        )


class GoogleGeminiLLMStrategy(ChatModelStrategy):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed")
        super().__init__(
            ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0.7,
                timeout=30,
                max_retries=2,
            ),
            model,
        )


class DockerLLMStrategy(ChatModelStrategy):
    def __init__(self, base_url: str, model: str = "phi3:mini"):
        from langchain_openai import ChatOpenAI

        super().__init__(
            ChatOpenAI(
                model=model,
                api_key="local-model",
                base_url=base_url,
                temperature=0.7,
                timeout=60,
                max_retries=1,
            ),
            model,
        )


class MockLLMStrategy(BaseLLMStrategy):
    """Deterministic offline mode used by tests and zero-key demos."""

    def generate_scenario(self, num_suspects: int = 4) -> CaseScenario:
        names = [
            ("Mara Voss", "Museum curator"),
            ("Elias Reed", "Security contractor"),
            ("Nina Park", "Investigative journalist"),
            ("Theo Grant", "Art restorer"),
            ("Iris Bell", "Gallery accountant"),
            ("Jonas Vale", "Private collector"),
        ]
        suspects = []
        for index, (name, occupation) in enumerate(names[:num_suspects], start=1):
            suspects.append(
                {
                    "id": f"s{index}",
                    "name": name,
                    "occupation": occupation,
                    "bio": (
                        f"{name} had authorized access to the gallery during the exhibition setup."
                    ),
                    "alibi": f"{name} says they left the gallery before 9:00 PM.",
                    "role": "criminal" if index == 2 else "suspect",
                }
            )
        return CaseScenario.model_validate(
            {
                "summary": (
                    "A prototype cipher device vanished from a locked gallery during a brief power "
                    "failure. Four people had access, but the security log and their timelines "
                    "disagree."
                ),
                "details": {
                    "crime": "Theft of a prototype cipher device",
                    "location": "Blackwood Gallery archive room",
                    "time_window": "9:05 PM-9:20 PM",
                    "clues": [
                        "The archive lock recorded a valid contractor badge at 9:12 PM.",
                        "The hallway camera lost power, but the archive lock did not.",
                        "Fresh graphite dust was found inside the device case.",
                    ],
                },
                "suspects": suspects,
                "criminal_id": "s2",
            }
        )

    def suspect_reply(
        self,
        suspect: dict[str, Any],
        scenario: dict[str, Any],
        question: str,
        chat_history: list[dict[str, str]],
    ) -> str:
        if chat_history:
            return (
                f"As I said earlier, {suspect['alibi']} About your question—{question}—"
                "I have nothing more to add without seeing the access log."
            )
        return f"{suspect['alibi']} I did not enter the archive during the outage."

    def analyze_evidence(
        self,
        scenario: dict[str, Any],
        suspect: dict[str, Any],
        last_answer: str,
        last_question: str,
        chat_history: list[dict[str, str]],
        current_score: float,
    ) -> EvidenceAnalysis:
        badge_clue = scenario["details"]["clues"][0]
        is_contractor = suspect.get("occupation") == "Security contractor"
        return EvidenceAnalysis(
            suspicion_delta=0.6 if is_contractor else 0.1,
            rationale=(
                "The stated timeline conflicts with the contractor badge record."
                if is_contractor
                else "The answer is broadly consistent, but the timeline remains unverified."
            ),
            contradiction_detected=is_contractor,
            contradiction=(
                "The suspect says they left before 9:00 PM, but their badge was used at 9:12 PM."
                if is_contractor
                else None
            ),
            relevant_clue=badge_clue if is_contractor else None,
        )


class LLMStrategyFactory:
    """Select a provider using explicit environment configuration."""

    @staticmethod
    def create_strategy() -> BaseLLMStrategy:
        local_url = os.environ.get("LOCAL_LLM_BASE_URL", "")
        if local_url:
            return DockerLLMStrategy(local_url, os.environ.get("LOCAL_LLM_MODEL", "phi3:mini"))

        google_key = os.environ.get("GOOGLE_API_KEY", "")
        if google_key and ChatGoogleGenerativeAI is not None:
            return GoogleGeminiLLMStrategy(
                google_key, os.environ.get("GOOGLE_MODEL", "gemini-2.5-flash-lite")
            )

        qwen_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY", "")
        if qwen_key:
            return QwenLLMStrategy(
                qwen_key,
                os.environ.get("QWEN_BASE_URL") or os.environ.get("DASHSCOPE_BASE_URL"),
                os.environ.get("QWEN_MODEL", "qwen-plus"),
            )

        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if openai_key:
            return OpenAILLMStrategy(openai_key, os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))

        return MockLLMStrategy()
