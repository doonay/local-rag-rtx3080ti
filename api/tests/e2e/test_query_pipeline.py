import os

import httpx
import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.getenv("RUN_E2E") != "1", reason="set RUN_E2E=1"),
]


def test_question_returns_sources() -> None:
    upload = httpx.post(
        "http://localhost:8000/api/v1/upload",
        files={
            "file": (
                "query-test.txt",
                "Париж — столица Франции. Эйфелева башня находится в Париже.".encode(),
                "text/plain",
            )
        },
        timeout=180,
    )
    assert upload.status_code == 201, upload.text

    response = httpx.post(
        "http://localhost:8000/api/v1/ask",
        json={"question": "Какая столица Франции?", "top_k": 3},
        timeout=300,
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"]
    assert body["sources"]
