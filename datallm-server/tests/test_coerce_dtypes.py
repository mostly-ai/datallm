import pytest
from datallm_server.coerce import coerce_datetime


def test_fix_invalid_days_with_valid_dates():
    assert coerce_datetime("2023-03-01") == "2023-03-01"
    assert coerce_datetime("2020-02-29") == "2020-02-29"  # Leap year test


def test_fix_invalid_days_with_invalid_dates():
    assert (
        coerce_datetime("2023-02-29") == "2023-02-28"
    )  # Non-leap year, Feb 29 should be adjusted to Feb 28
    assert coerce_datetime("2023-04-31") == "2023-04-30"  # April has 30 days
    assert coerce_datetime("2021-11-31") == "2021-11-30"  # November has 30 days


def test_fix_invalid_days_with_datetimes():
    assert (
        coerce_datetime("2023-02-29 10:00:00") == "2023-02-28 10:00:00"
    )  # Adjust date, keep time
    assert (
        coerce_datetime("2024-02-30 23:59:59") == "2024-02-29 23:59:59"
    )  # Leap year adjustment
