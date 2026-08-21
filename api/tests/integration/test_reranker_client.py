import os

import pytest

from api.services.reranker_client import RerankerClient


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1"),
]


async def test_reranker_orders_relevant_document_first() -> None:
    client = RerankerClient(base_url="http://localhost:8081")
    try:
        results = await client.rerank(
            "Столица Франции",
            ["Берлин — столица Германии", "Париж — столица Франции"],
            top_k=2,
        )
        assert results[0]["text"] == "Париж — столица Франции"
    finally:
        await client.close()
