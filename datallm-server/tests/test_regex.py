import re
from datallm_server.regex import convert_completion_dtype_to_regex
from datallm_server.types import DtypeEnum


def test_category_dtype():
    categories = ["cat1", "cat2", "cat3"]
    regex = convert_completion_dtype_to_regex(
        dtype=DtypeEnum.category, categories=categories
    )
    assert regex == "cat1|cat2|cat3"


def compile_and_match(regex: str, test_string: str) -> bool:
    return bool(re.match(f"^{regex}$", test_string))


def test_integer_dtype_matches():
    regex = convert_completion_dtype_to_regex(dtype=DtypeEnum.integer)
    assert compile_and_match(regex, "123")
    assert compile_and_match(regex, "-123")
    assert not compile_and_match(regex, "abc")
    assert not compile_and_match(regex, "NULL")
    assert not compile_and_match(regex, "")  # Empty string should not match


def test_float_dtype_matches():
    regex = convert_completion_dtype_to_regex(dtype=DtypeEnum.float)
    assert compile_and_match(regex, "123.456")
    assert compile_and_match(regex, "-123.456e+7")
    assert not compile_and_match(regex, "NULL")
    assert not compile_and_match(regex, "abc")


def test_boolean_dtype_matches():
    regex = convert_completion_dtype_to_regex(dtype=DtypeEnum.boolean)
    assert compile_and_match(regex, "True")
    assert compile_and_match(regex, "False")
    assert not compile_and_match(regex, "NULL")
    assert not compile_and_match(regex, "true")  # Case sensitivity check


def test_date_dtype_matches():
    regex = convert_completion_dtype_to_regex(dtype=DtypeEnum.date)
    assert compile_and_match(regex, "2020-12-31")
    assert not compile_and_match(regex, "NULL")
    assert not compile_and_match(regex, "2020-13-01")  # Invalid month


def test_datetime_dtype_matches():
    regex = convert_completion_dtype_to_regex(dtype=DtypeEnum.datetime)
    assert compile_and_match(regex, "2020-12-31 23:59:59")
    assert not compile_and_match(regex, "NULL")
    assert not compile_and_match(regex, "2020-12-31 24:00:00")  # Invalid hour


def test_category_dtype_matches():
    categories = ["cat1", "cat2", "cat3"]
    regex = convert_completion_dtype_to_regex(
        dtype=DtypeEnum.category, categories=categories
    )
    assert compile_and_match(regex, "cat1")
    assert not compile_and_match(regex, "NULL")
    assert not compile_and_match(regex, "cat4")
