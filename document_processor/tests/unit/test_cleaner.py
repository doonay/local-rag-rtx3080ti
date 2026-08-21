import pytest

from document_processor.cleaner import clean_text


pytestmark = pytest.mark.unit


def test_normalizes_spaces_and_unicode() -> None:
    assert clean_text("Cafe\u0301   resume") == "Café resume"


def test_removes_page_markers_and_rule_lines() -> None:
    text = "Page 1\nUseful text\n-----\nСтраница 2"
    assert clean_text(text) == "Useful text"


def test_joins_hyphenated_line_breaks() -> None:
    assert clean_text("docu-\nment") == "document"


def test_keeps_paragraph_boundaries() -> None:
    assert clean_text("First\n\n\n\nSecond") == "First\n\nSecond"
