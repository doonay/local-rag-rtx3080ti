import os
import uuid

import pytest

from api.services.embedder_client import HybridEmbedding, SparseVector
from api.services.qdrant_client import QdrantClient


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1"),
]


async def test_qdrant_upsert_and_search() -> None:
    collection = f"integration_{uuid.uuid4().hex}"
    client = QdrantClient(base_url="http://localhost:6333", collection=collection)
    point_id = str(uuid.uuid4())
    try:
        saved = await client.upsert_points(
            [
                {
                    "id": point_id,
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "sparse": {"indices": [10], "values": [1.0]},
                    },
                    "payload": {"text": "test point"},
                }
            ]
        )
        results = await client.search_hybrid(
            HybridEmbedding(
                dense=[1.0, 0.0, 0.0],
                sparse=SparseVector(indices=[10], values=[1.0]),
            ),
            limit=1,
        )
        assert saved is True
        assert results[0]["id"] == point_id
    finally:
        await client.client.delete(f"http://localhost:6333/collections/{collection}")
        await client.close()
