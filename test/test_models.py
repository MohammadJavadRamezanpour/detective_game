import pytest
from pydantic import ValidationError

from backend.llm_strategy import MockLLMStrategy
from backend.models import CaseScenario, EvidenceAnalysis


def test_mock_case_satisfies_structured_schema():
    scenario = MockLLMStrategy().generate_scenario(4)
    assert isinstance(scenario, CaseScenario)
    assert len(scenario.suspects) == 4
    assert [suspect.id for suspect in scenario.suspects] == ["s1", "s2", "s3", "s4"]


def test_case_rejects_mismatched_criminal():
    payload = MockLLMStrategy().generate_scenario(4).model_dump()
    payload["criminal_id"] = "s1"
    with pytest.raises(ValidationError, match="exactly one suspect"):
        CaseScenario.model_validate(payload)


def test_evidence_analysis_requires_contradiction_text():
    with pytest.raises(ValidationError, match="requires a description"):
        EvidenceAnalysis(
            suspicion_delta=0.2,
            rationale="The timelines differ.",
            contradiction_detected=True,
        )
