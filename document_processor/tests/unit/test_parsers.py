import pymupdf
import pytest
from fastapi.testclient import TestClient

from document_processor.main import app, decode_text, extract_text_from_pdf


pytestmark = pytest.mark.unit
client = TestClient(app)


def test_decodes_utf8_with_bom() -> None:
    assert decode_text("Привет".encode("utf-8-sig")) == "Привет"


def test_extracts_text_from_real_pdf() -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "A readable PDF page")
    content = document.tobytes()
    document.close()

    text, pages, method = extract_text_from_pdf(content)
    assert "readable PDF" in text
    assert pages == 1
    assert method == "pymupdf"


def test_processes_plain_text_document() -> None:
    response = client.post(
        "/process",
        files={"file": ("notes.txt", "Полезный текст документа для поиска.".encode(), "text/plain")},
        data={"chunk_size": "200", "overlap": "20", "strategy": "recursive"},
    )
    assert response.status_code == 200
    assert response.json()["chunks"] == 1
