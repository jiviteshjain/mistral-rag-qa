from datetime import datetime
from copy import deepcopy

import dateparser
from dateparser.search import search_dates
from pytz import timezone

SPACE = " "

EASTERN = timezone("US/Eastern")

DATE_PARSER_SETTINGS = {
    "DATE_ORDER": "MDY",
    "PREFER_LOCALE_DATE_ORDER": False,
    "TIMEZONE": "US/Eastern",
    "TO_TIMEZONE": "US/Eastern",
    "RETURN_AS_TIMEZONE_AWARE": True,
    "PREFER_DAY_OF_MONTH": "first",
    "PREFER_MONTH_OF_YEAR": "last",
    "RELATIVE_BASE": datetime.now(EASTERN),
}


def process(obj: str | list[str]) -> str:
    if isinstance(obj, str):
        return process_list([obj])
    elif isinstance(obj, list):
        return process_list(obj)
    else:
        raise ValueError(f"Invalid type: {type(obj)}")


# Use this for raw text, when the inner strings are not clean.
def process_list(obj: list[str]) -> str:
    # Normalize whitespace within each string as well.
    return join([join(x.strip().split()) for x in obj])


# Use this for faster processing of clean text,
# when the inner strings are already clean.
def join(obj: list[str]) -> str:
    return SPACE.join([x for x in obj if len(x) > 0])


def today() -> str:
    return datetime.now(EASTERN).isoformat()


def xpath_match_class(class_name: str) -> str:
    return f"contains(concat(' ', normalize-space(@class), ' '), ' {class_name} ')"


def parse_date(date: str, base: str | datetime | None = None) -> str:
    if isinstance(base, str):
        base = dateparser.parse(base, languages=["en"], settings=DATE_PARSER_SETTINGS)

    if base is not None:
        settings = deepcopy(DATE_PARSER_SETTINGS)
        settings["RELATIVE_BASE"] = base
    else:
        settings = DATE_PARSER_SETTINGS  # Not a deep copy, be careful.

    return dateparser.parse(
        date, languages=["en"], settings=settings
    ).isoformat()  # If it doesn't find a date and returns None, this will throw an error, which is desirable.


# Test before using.
def find_dates(text: str, base: datetime | None = None) -> list[str]:
    if isinstance(base, str):
        base = dateparser.parse(base, languages=["en"], settings=DATE_PARSER_SETTINGS)

    if base is not None:
        settings = deepcopy(DATE_PARSER_SETTINGS)
        settings["RELATIVE_BASE"] = base
    else:
        settings = DATE_PARSER_SETTINGS  # Not a deep copy, be careful.

    results = search_dates(text, languages=["en"], settings=settings)

    if results is None:
        return []

    return [x[1].isoformat() for x in results]
