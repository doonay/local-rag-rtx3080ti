from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_ask_requires_non_empty_question() -> None:
    response = client.post("/api/v1/ask", json={"question": ""})
    assert response.status_code == 422


def test_upload_requires_file() -> None:
    response = client.post("/api/v1/upload")
    assert response.status_code == 422


def test_upload_rejects_unsupported_file() -> None:
    response = client.post(
        "/api/v1/upload",
        files={"file": ("payload.exe", b"not a document", "application/octet-stream")},
    )
    assert response.status_code == 415
