import pytest

from datallm_server.types import RowCompletionsRequest, DtypeEnum


def test_max_tokens_with_string_dtype():
    RowCompletionsRequest(
        model="mistral-7b-datallm-v1",
        prompt="Test prompt",
        dtype=DtypeEnum.string,
        max_tokens=20,
    )
    assert True


# Test when dtype is not string and max_tokens is set, expecting a ValueError
def test_max_tokens_with_non_string_dtype():
    with pytest.raises(ValueError):
        RowCompletionsRequest(
            model="mistral-7b-datallm-v1",
            prompt="Test prompt",
            dtype=DtypeEnum.float,
            max_tokens=20,
        )


def test_no_max_tokens_with_non_string_dtype():
    RowCompletionsRequest(
        model="mistral-7b-datallm-v1", prompt="Test prompt", dtype=DtypeEnum.float
    )
    assert True


# Test when regex is set and dtype is not string, expecting a ValueError
def test_regex_with_non_string_dtype():
    with pytest.raises(ValueError):
        RowCompletionsRequest(
            model="mistral-7b-datallm-v1",
            prompt="Test prompt",
            dtype=DtypeEnum.category,
            regex="^Test.*",
        )


def test_no_regex_with_non_string_dtype():
    RowCompletionsRequest(
        model="mistral-7b-datallm-v1", prompt="Test prompt", dtype=DtypeEnum.float
    )
    assert True


# Test when dtype is category but categories are not set, expecting a ValueError
def test_category_dtype_without_categories():
    with pytest.raises(ValueError):
        RowCompletionsRequest(
            model="mistral-7b-datallm-v1",
            prompt="Test prompt",
            dtype=DtypeEnum.category,
        )


def test_category_dtype_with_categories():
    RowCompletionsRequest(
        model="mistral-7b-datallm-v1",
        prompt="Test prompt",
        dtype=DtypeEnum.category,
        categories=["cat1", "cat2"],
    )
    assert True
