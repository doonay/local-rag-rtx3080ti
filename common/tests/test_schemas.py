import pytest
from pydantic import ValidationError

from common.schemas import DocumentChunk, QueryRequest


pytestmark = pytest.mark.unit


def test_document_chunk_generates_id() -> None:
    chunk = DocumentChunk(text="Test", metadata={"source": "test.pdf"})
    assert chunk.id
    assert chunk.embedding is None


def test_document_chunk_rejects_empty_text() -> None:
    with pytest.raises(ValidationError):
        DocumentChunk(text="")


def test_query_top_k_is_bounded() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="Question", top_k=0)
