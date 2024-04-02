import interegular
import pytest

from datallm_server.modal_engine import EngineRequest
from datallm_server.types import (
    NamedValue,
    DtypeEnum,
    RowCompletionsRequest,
)


def test_conversion():
    row_completion_request = RowCompletionsRequest(
        model="mistral-7b-datallm-v1",
        prompt="Test prompt",
        data_description="Test context",
        rows=[[NamedValue(name="test_name", value="test_value")]],
        dtype=DtypeEnum.string,
        temperature=0.5,
        top_p=0.8,
        max_tokens=32,
        categories=["cat1", "cat2"],
    )

    # Convert to EngineRequest
    engine_request = EngineRequest.from_row_completion_request(row_completion_request)

    # Assertions
    assert "test_name" in engine_request.prompts[0]
    assert "test_value" in engine_request.prompts[0]
    assert engine_request.regex is None
    assert engine_request.max_tokens == 32
    assert engine_request.top_p == 0.8
    assert engine_request.temperature == 0.5


def test_raises_on_unsupported_regex():
    with pytest.raises(ValueError):
        row_completion_request = RowCompletionsRequest(
            model="mistral-7b-datallm-v1",
            prompt="Test prompt",
            data_description="Test context",
            rows=[[NamedValue(name="test_name", value="test_value")]],
            dtype=DtypeEnum.string,
            regex="^a",
        )
        EngineRequest.from_row_completion_request(row_completion_request)


def test_raises_on_invalid_regex():
    with pytest.raises(ValueError):
        row_completion_request = RowCompletionsRequest(
            model="mistral-7b-datallm-v1",
            prompt="Test prompt",
            data_description="Test context",
            rows=[[NamedValue(name="test_name", value="test_value")]],
            dtype=DtypeEnum.string,
            regex="[a-A]",
        )
        EngineRequest.from_row_completion_request(row_completion_request)
