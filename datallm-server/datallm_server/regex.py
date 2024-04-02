from datallm_server.types import DtypeEnum
import re


def convert_completion_dtype_to_regex(
    dtype: DtypeEnum, categories: list[str] | None = None
) -> str | None:
    if dtype == DtypeEnum.string:
        return None

    int_regex = r"[+-]?(0|[1-9][0-9]*)"  # int regex with +/- prefix
    float_regex = rf"{int_regex}(\.[0-9]+)?([eE][+-][0-9]+)?"  # float regex with scientific notation
    bool_regex = "(True|False)"
    date_regex = r"(19\d{2}|20\d{2})-(0[1-9]|1[0-2])-([0-2][0-9]|3[0-1])"  # date regex limited to years 1900-2099
    time_regex = r"([0-1][0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])"
    datetime_regex = rf"({date_regex}) ({time_regex})"

    if dtype == DtypeEnum.integer:
        return int_regex
    elif dtype == DtypeEnum.float:
        return float_regex
    elif dtype == DtypeEnum.boolean:
        return bool_regex
    elif dtype == DtypeEnum.date:
        return date_regex
    elif dtype == DtypeEnum.datetime:
        return datetime_regex
    elif dtype == DtypeEnum.category and categories:
        return r"|".join(re.escape(cat) for cat in categories)

    return None
