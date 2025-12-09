"""LLM Strategy Pattern Implementation

This module implements the Strategy design pattern for LLM operations,
allowing easy addition of multiple LLM providers while encapsulating all
LLM-related logic including scenario generation, suspect replies, and
suspicion analysis.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None


class BaseLLMStrategy(ABC):
    """Abstract base class defining the interface for all LLM strategies."""

    @abstractmethod
    def generate_scenario(self, num_suspects: int = 4) -> Dict[str, Any]:
        """Generate a deterministic scenario using the LLM.
        
        Args:
            num_suspects: Number of suspects to generate
            
        Returns:
            Dictionary containing scenario data with keys:
            - summary: Brief case summary
            - details: Crime details (crime, location, time_window, clues)
            - suspects: List of suspect dictionaries
            - criminal_id: ID of the criminal suspect
        """
        pass

    @abstractmethod
    def suspect_reply(
        self,
        suspect: Dict[str, Any],
        scenario: Dict[str, Any],
        question: str,
        chat_history: List[Any],
    ) -> str:
        """Generate a suspect's reply guided by persona and scenario.
        
        Args:
            suspect: Suspect data (id, name, bio, alibi, role)
            scenario: Scenario data (summary, details)
            question: Player's question
            chat_history: Previous conversation messages
            
        Returns:
            Suspect's response as a string
        """
        pass

    @abstractmethod
    def analyze_suspicion(
        self,
        scenario: Dict[str, Any],
        suspect: Dict[str, Any],
        last_answer: str,
        last_question: str,
        current_score: float,
    ) -> float:
        """Analyze suspicion level based on the latest exchange.
        
        Args:
            scenario: Scenario data
            suspect: Suspect data
            last_answer: Suspect's last answer
            last_question: Player's last question
            current_score: Current suspicion score
            
        Returns:
            Delta to adjust suspicion score (typically -0.5 to 0.8)
        """
        pass

    @abstractmethod
    def invoke(self, messages: List[Any]) -> Any:
        """Low-level LLM invocation for backward compatibility.
        
        Args:
            messages: List of messages to send to the LLM
            
        Returns:
            LLM response object
        """
        pass


class QwenLLMStrategy(BaseLLMStrategy):
    """Strategy for Qwen/DashScope LLM."""

    def __init__(self, api_key: str, base_url: str = None, model: str = None):
        """Initialize Qwen strategy.
        
        Args:
            api_key: Qwen/DashScope API key
            base_url: Optional base URL for API endpoint
            model: Optional model name (default: qwen-plus)
        """
        from langchain_openai import ChatOpenAI

        self.base_url = base_url or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.model = model or "qwen-plus"
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0.7
        )

    def generate_scenario(self, num_suspects: int = 4) -> Dict[str, Any]:
        """Generate scenario using Qwen LLM."""
        # Use deterministic generation for planning
        deterministic = self.llm.bind(temperature=0.0)

        sys = SystemMessage(
            content=(
                "You are generating a grounded detective interrogation case for a web game. "
                "Output STRICT JSON ONLY (no markdown, no commentary). Schema:\n"
                '{\n  "summary": string,\n  "details": {\n    "crime": string,\n    "location": string,\n    "time_window": string,\n    "clues": [string, ...]\n  },\n  "suspects": [\n    { "id": \'s1\', "name": string, "occupation": string, "bio": string, "alibi": string, "role": \'suspect\'|\'criminal\' }\n  ],\n  "criminal_id": \'sX\'\n}\n'
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

    def suspect_reply(
        self,
        suspect: Dict[str, Any],
        scenario: Dict[str, Any],
        question: str,
        chat_history: List[Any],
    ) -> str:
        """Generate suspect reply using Qwen LLM."""
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
        resp = self.llm.invoke(messages)
        return getattr(resp, "content", str(resp))

    def analyze_suspicion(
        self,
        scenario: Dict[str, Any],
        suspect: Dict[str, Any],
        last_answer: str,
        last_question: str,
        current_score: float,
    ) -> float:
        """Analyze suspicion using Qwen LLM."""
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
            resp = self.llm.invoke([sys, hm])
            text = getattr(resp, "content", "0.0").strip()
            # Extract the first numeric value
            m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            if m:
                val = float(m.group(0))
                return float(max(min(val, 0.8), -0.5))
        except Exception:
            pass
        return 0.0

    def invoke(self, messages: List[Any]) -> Any:
        """Invoke Qwen LLM directly."""
        return self.llm.invoke(messages)


class OpenAILLMStrategy(BaseLLMStrategy):
    """Strategy for OpenAI GPT models."""

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini"):
        """Initialize OpenAI strategy.
        
        Args:
            api_key: Optional OpenAI API key (uses env var if not provided)
            model: Model name (default: gpt-4o-mini)
        """
        from langchain_openai import ChatOpenAI

        self.model = model
        if api_key:
            self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=0.7)
        else:
            self.llm = ChatOpenAI(model=model, temperature=0.7)

    def generate_scenario(self, num_suspects: int = 4) -> Dict[str, Any]:
        """Generate scenario using OpenAI LLM."""
        # Use deterministic generation for planning
        deterministic = self.llm.bind(temperature=0.0)

        sys = SystemMessage(
            content=(
                "You are generating a grounded detective interrogation case for a web game. "
                "Output STRICT JSON ONLY (no markdown, no commentary). Schema:\n"
                '{\n  "summary": string,\n  "details": {\n    "crime": string,\n    "location": string,\n    "time_window": string,\n    "clues": [string, ...]\n  },\n  "suspects": [\n    { "id": \'s1\', "name": string, "occupation": string, "bio": string, "alibi": string, "role": \'suspect\'|\'criminal\' }\n  ],\n  "criminal_id": \'sX\'\n}\n'
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

    def suspect_reply(
        self,
        suspect: Dict[str, Any],
        scenario: Dict[str, Any],
        question: str,
        chat_history: List[Any],
    ) -> str:
        """Generate suspect reply using OpenAI LLM."""
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
        resp = self.llm.invoke(messages)
        return getattr(resp, "content", str(resp))

    def analyze_suspicion(
        self,
        scenario: Dict[str, Any],
        suspect: Dict[str, Any],
        last_answer: str,
        last_question: str,
        current_score: float,
    ) -> float:
        """Analyze suspicion using OpenAI LLM."""
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
            resp = self.llm.invoke([sys, hm])
            text = getattr(resp, "content", "0.0").strip()
            # Extract the first numeric value
            m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            if m:
                val = float(m.group(0))
                return float(max(min(val, 0.8), -0.5))
        except Exception:
            pass
        return 0.0

    def invoke(self, messages: List[Any]) -> Any:
        """Invoke OpenAI LLM directly."""
        return self.llm.invoke(messages)


class GoogleGeminiLLMStrategy(BaseLLMStrategy):
    """Strategy for Google Gemini models."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        """Initialize Google Gemini strategy.
        
        Args:
            api_key: Google API key
            model: Model name (default: gemini-2.0-flash-exp)
        """
        if not ChatGoogleGenerativeAI:
            raise ImportError("langchain-google-genai package not installed")

        self.model = model
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.7,
            convert_system_message_to_human=True
        )

    def generate_scenario(self, num_suspects: int = 4) -> Dict[str, Any]:
        """Generate scenario using Google Gemini LLM."""
        # Use deterministic generation for planning
        deterministic = self.llm.bind(temperature=0.0)

        sys = SystemMessage(
            content=(
                "You are generating a grounded detective interrogation case for a web game. "
                "Output STRICT JSON ONLY (no markdown, no commentary). Schema:\n"
                '{\n  "summary": string,\n  "details": {\n    "crime": string,\n    "location": string,\n    "time_window": string,\n    "clues": [string, ...]\n  },\n  "suspects": [\n    { "id": \'s1\', "name": string, "occupation": string, "bio": string, "alibi": string, "role": \'suspect\'|\'criminal\' }\n  ],\n  "criminal_id": \'sX\'\n}\n'
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

    def suspect_reply(
        self,
        suspect: Dict[str, Any],
        scenario: Dict[str, Any],
        question: str,
        chat_history: List[Any],
    ) -> str:
        """Generate suspect reply using Google Gemini LLM."""
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
        resp = self.llm.invoke(messages)
        return getattr(resp, "content", str(resp))

    def analyze_suspicion(
        self,
        scenario: Dict[str, Any],
        suspect: Dict[str, Any],
        last_answer: str,
        last_question: str,
        current_score: float,
    ) -> float:
        """Analyze suspicion using Google Gemini LLM."""
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
            resp = self.llm.invoke([sys, hm])
            text = getattr(resp, "content", "0.0").strip()
            # Extract the first numeric value
            m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            if m:
                val = float(m.group(0))
                return float(max(min(val, 0.8), -0.5))
        except Exception:
            pass
        return 0.0

    def invoke(self, messages: List[Any]) -> Any:
        """Invoke Google Gemini LLM directly."""
        return self.llm.invoke(messages)


class DockerLLMStrategy(BaseLLMStrategy):
    """Strategy for local Docker-based LLMs (e.g., Ollama, vLLM)."""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "phi3:mini"):
        """
        Args:
            base_url: The base URL for the local model API (OpenAI compatible)
            model: Local model name (must exist inside the Docker container)
        """
        from langchain_openai import ChatOpenAI

        self.base_url = base_url
        self.model = model

        # No API key for local LLM
        self.llm = ChatOpenAI(
            model=self.model,
            api_key="not-needed",
            base_url=self.base_url,
            temperature=0.7
        )

    # ---- REUSE SAME PROMPTS AS OTHER STRATEGIES -----

    def generate_scenario(self, num_suspects: int = 4) -> Dict[str, Any]:
        deterministic = self.llm.bind(temperature=0.0)

        sys = SystemMessage(
            content=(
                "You are generating a grounded detective interrogation case for a web game. "
                "Output STRICT JSON ONLY (no markdown, no commentary). Schema:\n"
                '{\n  "summary": string,\n  "details": {\n    "crime": string,\n    "location": string,\n    "time_window": string,\n    "clues": [string, ...]\n  },\n  "suspects": [\n    { "id": \'s1\', "name": string, "occupation": string, "bio": string, "alibi": string, "role": \'suspect\'|\'criminal\' }\n  ],\n  "criminal_id": \'sX\'\n}\n'
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
        raw = resp.content.strip()
        raw = re.sub(r"^```(json)?\n|\n```$", "", raw)
        data = json.loads(raw)

        # Normalization (same as other strategies)
        suspects = data["suspects"]
        for i, s in enumerate(suspects, start=1):
            s["id"] = f"s{i}"
            s.setdefault("role", "suspect")

        return {
            "summary": data["summary"],
            "details": data["details"],
            "suspects": suspects,
            "criminal_id": data.get("criminal_id", "s1")
        }

    def suspect_reply(self, suspect, scenario, question, chat_history):
        role = suspect.get("role", "suspect")
        persona = suspect.get("bio", "")
        alibi = suspect.get("alibi", "")
        name = suspect.get("name", "Suspect")

        system_prompt = (
            "You are role-playing as a suspect in an interrogation game. "
            "Stay in character, use first person, and defend yourself. "
            "Keep responses concise (2–5 sentences). "
            f"Your name: {name}. Persona: {persona}. Alibi: {alibi}. "
            f"Case summary: {scenario.get('summary','')}."
        )

        if role == "criminal":
            system_prompt += " As the criminal, be evasive but not obviously contradictory."
        else:
            system_prompt += " As an innocent suspect, be consistent and cooperative."

        resp = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ])

        return resp.content

    def analyze_suspicion(self, scenario, suspect, last_answer, last_question, current_score):
        sys = SystemMessage(
            content=(
                "Score the suspicion of the suspect's answer. "
                "Return ONLY a float delta between -0.5 and 0.8."
            )
        )

        hm = HumanMessage(
            content=(
                f"Question: {last_question}\n"
                f"Answer: {last_answer}\n"
                f"Persona: {suspect.get('bio','')}\n"
                f"Scenario: {scenario.get('summary','')}\n"
                f"Current: {current_score}"
            )
        )

        try:
            resp = self.llm.invoke([sys, hm])
            txt = resp.content.strip()
            m = re.search(r"-?\d+(\.\d+)?", txt)
            if m:
                return max(-0.5, min(0.8, float(m.group())))
        except Exception:
            pass
        return 0.0

    def invoke(self, messages: List[Any]):
        return self.llm.invoke(messages)


class MockLLMStrategy(BaseLLMStrategy):
    """Mock LLM strategy for testing without API keys."""

    def generate_scenario(self, num_suspects: int = 4) -> Dict[str, Any]:
        """Generate a mock scenario."""
        suspects = []
        for i in range(1, num_suspects + 1):
            suspects.append({
                "id": f"s{i}",
                "name": f"Suspect {i}",
                "occupation": f"Occupation {i}",
                "bio": f"This is suspect {i}'s background story.",
                "alibi": f"I was at location {i} during the crime.",
                "role": "criminal" if i == 1 else "suspect"
            })

        return {
            "summary": "A mock crime has occurred for testing purposes.",
            "details": {
                "crime": "Mock theft",
                "location": "Mock location",
                "time_window": "Mock time",
                "clues": ["Mock clue 1", "Mock clue 2"]
            },
            "suspects": suspects,
            "criminal_id": "s1"
        }

    def suspect_reply(
        self,
        suspect: Dict[str, Any],
        scenario: Dict[str, Any],
        question: str,
        chat_history: List[Any],
    ) -> str:
        """Generate a mock suspect reply."""
        name = suspect.get("name", "Suspect")
        alibi = suspect.get("alibi", "I don't recall")
        return (
            f"[Mock Response from {name}] "
            f"I hear your question: '{question}'. "
            f"{alibi} I don't recall anything suspicious. "
            f"I was busy with my own tasks."
        )

    def analyze_suspicion(
        self,
        scenario: Dict[str, Any],
        suspect: Dict[str, Any],
        last_answer: str,
        last_question: str,
        current_score: float,
    ) -> float:
        """Analyze suspicion using simple heuristics."""
        delta = 0.0
        lower = (last_answer or "").lower()
        # Crude signals
        suspicious_markers = ["avoid", "confuse", "contrad", "maybe", "think", "unsure", "don't recall", "forgot"]
        for w in suspicious_markers:
            if w in lower:
                delta += 0.3
        if "alibi" in lower and ("changed" in lower or "different" in lower):
            delta += 0.5
        return min(max(delta, -0.2), 0.8)

    def invoke(self, messages: List[Any]) -> Any:
        """Mock LLM invocation."""
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


class LLMStrategyFactory:
    """Factory class to create appropriate LLM strategy based on available API keys."""

    @staticmethod
    def _get_local_llm_url() -> str:
        return os.environ.get("LOCAL_LLM_BASE_URL", "")

    @staticmethod
    def _get_qwen_key() -> str:
        """Return Qwen/DashScope API key if present."""
        return os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or ""

    @staticmethod
    def _get_openai_key() -> str:
        """Return OpenAI API key if present."""
        return os.environ.get("OPENAI_API_KEY") or ""

    @staticmethod
    def _get_google_key() -> str:
        """Return Google API key if present."""
        return os.environ.get("GOOGLE_API_KEY") or ""

    @staticmethod
    def create_strategy() -> BaseLLMStrategy:
        """Create appropriate LLM strategy based on available API keys.
        
        Preference order:
        1. Google Gemini if GOOGLE_API_KEY is set
        2. Qwen if DASHSCOPE_API_KEY or QWEN_API_KEY is set
        3. OpenAI if OPENAI_API_KEY is set
        4. MockLLMStrategy otherwise
        
        Returns:
            An instance of BaseLLMStrategy
        """
        # Prefer Google Gemini if key is available
        google_key = LLMStrategyFactory._get_google_key()
        if google_key and ChatGoogleGenerativeAI:
            print("Using Google Gemini LLM Strategy")
            return GoogleGeminiLLMStrategy(api_key=google_key)

        # Prefer Qwen if a Qwen/DashScope key is available
        qwen_key = LLMStrategyFactory._get_qwen_key()
        if qwen_key:
            base_url = (
                os.environ.get("QWEN_BASE_URL")
                or os.environ.get("DASHSCOPE_BASE_URL")
                or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
            )
            model = os.environ.get("QWEN_MODEL", "qwen-plus")
            print(f"Using Qwen LLM Strategy (model: {model})")
            try:
                return QwenLLMStrategy(api_key=qwen_key, base_url=base_url, model=model)
            except Exception as e:
                print(f"Failed to initialize Qwen strategy: {e}")
                # Fall through to next option

        # Otherwise fall back to OpenAI if available
        openai_key = LLMStrategyFactory._get_openai_key()
        if openai_key:
            print("Using OpenAI LLM Strategy")
            try:
                return OpenAILLMStrategy(api_key=openai_key)
            except Exception as e:
                print(f"Failed to initialize OpenAI strategy: {e}")
                # Fall through to mock

        # No keys found, use mock
        print("Using Mock LLM Strategy (no API keys found)")
        return MockLLMStrategy()
