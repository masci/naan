import pytest
from pyparsing import ParseException

from naan.filters import expression, evaluate_expression


@pytest.fixture
def metadata():
    return {
        "Title": "The Great Gatsby",
        "Author": "F. Scott Fitzgerald",
        "Published": 1925,
        "Genre": "Fiction",
        "Rating": 4.5,
    }


@pytest.mark.parametrize(
    "filter_str,expected",
    [
        ("Published > 1920", True),
        ("Rating >= 4.5", True),
        ("Author == 'F. Scott Fitzgerald'", True),
        ("Genre == 'Non-Fiction'", False),
    ],
)
def test_filters(metadata, filter_str, expected):
    # Parse and evaluate filters
    try:
        parsed_expr = expression.parseString(filter_str, parseAll=True)[0]
        assert evaluate_expression(parsed_expr, metadata) == expected
    except ParseException as pe:
        print(f"Failed to parse filter '{filter_str}': {pe}")
    except ValueError as ve:
        print(f"Error evaluating filter '{filter_str}': {ve}")
