from typing import List

import pandas as pd
import pyarrow as pa

from datallm._types import DtypeEnum

DTYPE_PANDAS_MAP = {
    DtypeEnum.integer: "int64[pyarrow]",
    DtypeEnum.float: "float64[pyarrow]",
    DtypeEnum.boolean: "bool[pyarrow]",  # bool_coerce
    DtypeEnum.category: "category",
    DtypeEnum.string: "string[pyarrow]",
    DtypeEnum.date: "datetime64[s]",  # pd.DatetimeIndex,
    DtypeEnum.datetime: "datetime64[s]",  # pd.DatetimeIndex
}
DTYPE_PA_CAST_FUNC = {
    DtypeEnum.integer: pa.int64(),
    DtypeEnum.float: pa.float64(),
    DtypeEnum.boolean: pa.bool_(),
}


def _convert_values_to_series(
    values: List[str],
    dtype: DtypeEnum,
) -> pd.Series:
    pandas_dtype = DTYPE_PANDAS_MAP[dtype]
    pa_cast_func = DTYPE_PA_CAST_FUNC.get(dtype)
    if pa_cast_func:
        values = pa.compute.cast(values, pa_cast_func)
    if dtype in [DtypeEnum.date, DtypeEnum.datetime]:
        series = pd.to_datetime(pd.Series(values), errors="coerce")
    else:
        series = pd.Series(values, dtype=pandas_dtype)
    return series
