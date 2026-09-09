from unittest.mock import patch

from backend.llm_strategy import LLMStrategyFactory, MockLLMStrategy


def test_factory_uses_offline_mode_without_configuration():
    with patch.dict("os.environ", {}, clear=True):
        assert isinstance(LLMStrategyFactory.create_strategy(), MockLLMStrategy)


def test_factory_selects_openai():
    with (
        patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True),
        patch("backend.llm_strategy.OpenAILLMStrategy") as strategy,
    ):
        LLMStrategyFactory.create_strategy()
        strategy.assert_called_once_with("test-key", "gpt-4o-mini")


def test_factory_prefers_local_model():
    environment = {
        "LOCAL_LLM_BASE_URL": "http://localhost:11434/v1",
        "OPENAI_API_KEY": "test-key",
    }
    with (
        patch.dict("os.environ", environment, clear=True),
        patch("backend.llm_strategy.DockerLLMStrategy") as strategy,
    ):
        LLMStrategyFactory.create_strategy()
        strategy.assert_called_once_with("http://localhost:11434/v1", "phi3:mini")
