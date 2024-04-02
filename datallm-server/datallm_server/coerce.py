import calendar
from datetime import datetime
from typing import List

from datallm_server.types import DtypeEnum


def coerce_datetime(text: str) -> str:
    """
    Ensure that the text is a valid date or datetime string.
    """
    # extract year, month, and day from the ISO formatted text
    y, m, d = int(text[:4]), int(text[5:7]), int(text[8:10])
    # set to last day of month, in case of too large day value
    last_day = calendar.monthrange(y, m)[1]
    d = min(d, last_day)
    dt_str = f"{y:04d}-{m:02d}-{d:02d}" + text[10:]
    # convert to date and back to check for valid date
    dt_str = datetime.fromisoformat(dt_str).isoformat().replace("T", " ")
    # trim to original length
    dt_str = dt_str[: len(text)]
    return dt_str


def coerce_int(text: str) -> str:
    return str(int(float(text)))


def coerce_float(text: str) -> str:
    return str(float(text))


def coerce_dtype(values: List[str], dtype: DtypeEnum) -> List[str]:
    """Ensure that the strings can be converted to the requested dtype"""
    if dtype in [DtypeEnum.date, DtypeEnum.datetime]:
        coerce_fn = coerce_datetime
    elif dtype == DtypeEnum.integer:
        coerce_fn = coerce_int
    elif dtype == DtypeEnum.float:
        coerce_fn = coerce_float
    else:
        coerce_fn = lambda x: x
    if coerce_fn:
        values = [coerce_fn(v) for v in values]
    return values
