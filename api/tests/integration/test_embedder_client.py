import os

import pytest

from api.services.embedder_client import EmbedderClient


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1"),
]


async def test_embedder_health_and_vector() -> None:
    client = EmbedderClient(base_url="http://localhost:8080")
    try:
        health = await client.health_check()
        vector = await client.embed_single("Integration test document")
        assert health["status"] == "healthy"
        assert len(vector.dense) == health["dimension"]
        assert vector.sparse.indices
    finally:
        await client.close()
