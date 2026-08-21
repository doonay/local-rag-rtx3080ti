import pytest

from reranker_service.main import format_instruction


pytestmark = pytest.mark.unit


def test_formats_query_document_and_instruction() -> None:
    value = format_instruction("Find an answer", "Question", "Document")
    assert "<Instruct>: Find an answer" in value
    assert "<Query>: Question" in value
    assert "<Document>: Document" in value
