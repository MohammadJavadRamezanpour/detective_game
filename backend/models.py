"""Validated domain models shared by the workflow and LLM providers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class CrimeDetails(BaseModel):
    crime: str = Field(min_length=3, max_length=200)
    location: str = Field(min_length=2, max_length=200)
    time_window: str = Field(min_length=2, max_length=120)
    clues: list[str] = Field(min_length=2, max_length=8)


class Suspect(BaseModel):
    id: str = Field(pattern=r"^s\d+$")
    name: str = Field(min_length=2, max_length=80)
    occupation: str = Field(min_length=2, max_length=100)
    bio: str = Field(min_length=10, max_length=500)
    alibi: str = Field(min_length=5, max_length=500)
    role: Literal["suspect", "criminal"]


class CaseScenario(BaseModel):
    summary: str = Field(min_length=20, max_length=800)
    details: CrimeDetails
    suspects: list[Suspect] = Field(min_length=3, max_length=6)
    criminal_id: str = Field(pattern=r"^s\d+$")

    @model_validator(mode="after")
    def validate_criminal(self) -> CaseScenario:
        ids = [suspect.id for suspect in self.suspects]
        if len(ids) != len(set(ids)):
            raise ValueError("suspect IDs must be unique")

        criminals = [suspect.id for suspect in self.suspects if suspect.role == "criminal"]
        if criminals != [self.criminal_id]:
            raise ValueError("exactly one suspect must match criminal_id")
        return self


class EvidenceAnalysis(BaseModel):
    """A grounded, display-safe assessment of one interrogation answer."""

    suspicion_delta: float = Field(ge=-0.5, le=0.8)
    rationale: str = Field(min_length=3, max_length=240)
    contradiction_detected: bool = False
    contradiction: str | None = Field(default=None, max_length=240)
    relevant_clue: str | None = Field(default=None, max_length=240)

    @model_validator(mode="after")
    def keep_contradiction_consistent(self) -> EvidenceAnalysis:
        if not self.contradiction_detected:
            self.contradiction = None
        elif not self.contradiction:
            raise ValueError("a detected contradiction requires a description")
        return self
