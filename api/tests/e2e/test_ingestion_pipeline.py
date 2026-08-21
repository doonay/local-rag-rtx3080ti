import os

import httpx
import pytest


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.getenv("RUN_E2E") != "1", reason="set RUN_E2E=1"),
]


def test_upload_document() -> None:
    response = httpx.post(
        "http://localhost:8000/api/v1/upload",
        files={
            "file": (
                "integration.txt",
                "Париж — столица Франции. Эйфелева башня находится в Париже.".encode(),
                "text/plain",
            )
        },
        timeout=180,
    )
    assert response.status_code == 201, response.text
    assert response.json()["chunks_uploaded"] >= 1
