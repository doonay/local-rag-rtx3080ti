import uuid

import pytest

from local_app.embeddings import HybridEmbedding, SparseVector
from local_app.store import LocalVectorStore


pytestmark = pytest.mark.unit


def test_local_qdrant_hybrid_lifecycle(tmp_path) -> None:
    store = LocalVectorStore(tmp_path / "qdrant")
    document_id = str(uuid.uuid4())
    point_id = str(uuid.uuid4())
    try:
        store.upsert_points(
            [
                {
                    "id": point_id,
                    "vector": {
                        "dense": [1.0, 0.0, 0.0],
                        "sparse": {"indices": [10], "values": [1.0]},
                    },
                    "payload": {
                        "text": "Paris is the capital of France",
                        "document_id": document_id,
                        "filename": "test.txt",
                        "total_chunks": 1,
                        "content_hash": "abc123",
                    },
                }
            ]
        )
        results = store.search_hybrid(
            HybridEmbedding(
                dense=[1.0, 0.0, 0.0],
                sparse=SparseVector(indices=[10], values=[1.0]),
            ),
            limit=1,
        )
        assert results[0].id == point_id
        assert store.find_document_by_hash("abc123")["document_id"] == document_id
        assert store.list_documents()[0]["filename"] == "test.txt"

        store.delete_document(document_id)
        assert store.list_documents() == []
    finally:
        store.close()
