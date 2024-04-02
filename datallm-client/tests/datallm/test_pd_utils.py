import pandas as pd
import pytest

from datallm import DtypeEnum
from datallm._pd_utils import _convert_values_to_series


@pytest.mark.parametrize(
    "dtype,values,expected_result,expected_dtype",
    [
        (DtypeEnum.integer, ["1", "-2", "3"], [1, -2, 3], "int64[pyarrow]"),
        (DtypeEnum.float, ["1.2", "-2.3", "4.0"], [1.2, -2.3, 4], "float64[pyarrow]"),
        (DtypeEnum.boolean, ["True", "False"], [True, False], "bool[pyarrow]"),
        (DtypeEnum.category, ["a", "a", "b"], ["a", "a", "b"], "category"),
        (DtypeEnum.string, ["s", "t", "r"], ["s", "t", "r"], "string[pyarrow]"),
        (
            DtypeEnum.date,
            ["2021-01-01", "2021-01-02", "2021-01-03"],
            ["2021-01-01", "2021-01-02", "2021-01-03"],
            "datetime64[ns]",
        ),
        (
            DtypeEnum.datetime,
            ["2001-04-05 01:23", "2002-01-02 4:56"],
            ["2001-04-05 01:23", "2002-01-02 4:56"],
            "datetime64[ns]",
        ),
    ],
)
def test_convert_values_to_series(dtype, values, expected_result, expected_dtype):
    actual_series = _convert_values_to_series(values, dtype)
    expected_series = pd.Series(expected_result, dtype=expected_dtype)
    pd.testing.assert_series_equal(actual_series, expected_series)
