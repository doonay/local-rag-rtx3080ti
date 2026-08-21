import pytest

from document_processor.chunker import DocumentChunker, chunk_text


pytestmark = pytest.mark.unit


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("Short text", chunk_size=100, overlap=10) == ["Short text"]


def test_recursive_chunks_respect_size() -> None:
    chunks = DocumentChunker(chunk_size=40, overlap=8).chunk_text("word " * 40)
    assert len(chunks) > 1
    assert all(len(chunk["text"]) <= 40 for chunk in chunks)


def test_sentence_strategy_preserves_content() -> None:
    text = "First sentence. Second sentence. Third sentence."
    chunks = DocumentChunker(chunk_size=35, overlap=0).chunk_text(text, "sentence")
    assert "First sentence." in chunks[0]["text"]
    assert "Third sentence." in chunks[-1]["text"]


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        DocumentChunker(chunk_size=100, overlap=100)
