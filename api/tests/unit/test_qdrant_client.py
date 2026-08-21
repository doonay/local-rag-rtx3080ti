import pytest

from api.services.embedder_client import HybridEmbedding, SparseVector
from api.services.qdrant_client import QdrantClient


pytestmark = pytest.mark.unit


def test_builds_server_side_rrf_query() -> None:
    embedding = HybridEmbedding(
        dense=[0.1, 0.2, 0.3],
        sparse=SparseVector(indices=[11, 42], values=[0.7, 0.4]),
    )
    payload = QdrantClient.build_hybrid_query(embedding, limit=20)
    assert payload["query"] == {"fusion": "rrf"}
    assert payload["prefetch"][0]["using"] == "dense"
    assert payload["prefetch"][1]["using"] == "sparse"
    assert payload["prefetch"][1]["query"]["indices"] == [11, 42]
