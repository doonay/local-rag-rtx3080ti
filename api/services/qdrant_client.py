import os
from typing import Any, Sequence

import httpx

from .embedder_client import HybridEmbedding


class QdrantClient:
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(
        self,
        base_url: str | None = None,
        collection: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.getenv("QDRANT_URL", "http://qdrant:6333")).rstrip("/")
        self.collection = collection or os.getenv("QDRANT_COLLECTION", "documents")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def ensure_collection(self, vector_size: int) -> None:
        endpoint = f"{self.base_url}/collections/{self.collection}"
        response = await self.client.get(endpoint)
        if response.status_code == 200:
            vectors = response.json()["result"]["config"]["params"]["vectors"]
            dense_config = vectors.get(self.DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
            configured_size = dense_config.get("size") if isinstance(dense_config, dict) else None
            if configured_size != vector_size:
                raise RuntimeError(
                    "Qdrant collection does not match the BGE-M3 hybrid schema. "
                    "Delete the Qdrant volume and re-index the documents."
                )
            return
        if response.status_code != 404:
            response.raise_for_status()

        response = await self.client.put(
            endpoint,
            json={
                "vectors": {
                    self.DENSE_VECTOR_NAME: {"size": vector_size, "distance": "Cosine"}
                },
                "sparse_vectors": {
                    self.SPARSE_VECTOR_NAME: {"index": {"on_disk": False}}
                },
            },
        )
        response.raise_for_status()

        for field_name, field_schema in (
            ("document_id", "keyword"),
            ("filename", "keyword"),
            ("content_hash", "keyword"),
        ):
            index_response = await self.client.put(
                f"{endpoint}/index",
                params={"wait": "true"},
                json={"field_name": field_name, "field_schema": field_schema},
            )
            index_response.raise_for_status()

    async def collection_exists(self) -> bool:
        response = await self.client.get(f"{self.base_url}/collections/{self.collection}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True

    async def find_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        if not await self.collection_exists():
            return None
        response = await self.client.post(
            f"{self.base_url}/collections/{self.collection}/points/scroll",
            json={
                "limit": 1,
                "with_payload": True,
                "with_vector": False,
                "filter": {
                    "must": [{"key": "content_hash", "match": {"value": content_hash}}]
                },
            },
        )
        response.raise_for_status()
        points = response.json().get("result", {}).get("points", [])
        return points[0].get("payload", {}) if points else None

    async def list_documents(self) -> list[dict[str, Any]]:
        if not await self.collection_exists():
            return []
        documents: dict[str, dict[str, Any]] = {}
        offset: str | int | None = None
        while True:
            payload: dict[str, Any] = {
                "limit": 256,
                "with_payload": True,
                "with_vector": False,
            }
            if offset is not None:
                payload["offset"] = offset
            response = await self.client.post(
                f"{self.base_url}/collections/{self.collection}/points/scroll",
                json=payload,
            )
            response.raise_for_status()
            result = response.json().get("result", {})
            for point in result.get("points", []):
                point_payload = point.get("payload", {})
                document_id = point_payload.get("document_id")
                if document_id and document_id not in documents:
                    documents[document_id] = {
                        "document_id": document_id,
                        "filename": point_payload.get("filename", "unknown"),
                        "chunks": point_payload.get("total_chunks", 0),
                        "pages": point_payload.get("pages", 1),
                        "uploaded_at": point_payload.get("uploaded_at"),
                        "content_hash": point_payload.get("content_hash"),
                    }
            offset = result.get("next_page_offset")
            if offset is None:
                break
        return sorted(
            documents.values(),
            key=lambda document: document.get("uploaded_at") or "",
            reverse=True,
        )

    async def upsert_points(self, points: Sequence[dict[str, Any]]) -> bool:
        if not points:
            return False
        vectors = points[0].get("vector", {})
        dense = vectors.get(self.DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
        if not isinstance(dense, list) or not dense:
            raise ValueError("Each Qdrant point must contain a named dense vector")
        await self.ensure_collection(len(dense))
        response = await self.client.put(
            f"{self.base_url}/collections/{self.collection}/points",
            params={"wait": "true"},
            json={"points": list(points)},
        )
        response.raise_for_status()
        return response.json().get("status") == "ok"

    @classmethod
    def build_hybrid_query(cls, embedding: HybridEmbedding, limit: int) -> dict[str, Any]:
        prefetch: list[dict[str, Any]] = [
            {
                "query": embedding.dense,
                "using": cls.DENSE_VECTOR_NAME,
                "limit": limit,
            }
        ]
        if embedding.sparse.indices:
            prefetch.append(
                {
                    "query": {
                        "indices": embedding.sparse.indices,
                        "values": embedding.sparse.values,
                    },
                    "using": cls.SPARSE_VECTOR_NAME,
                    "limit": limit,
                }
            )
        if len(prefetch) == 1:
            return {
                "query": embedding.dense,
                "using": cls.DENSE_VECTOR_NAME,
                "limit": limit,
                "with_payload": True,
            }
        return {
            "prefetch": prefetch,
            "query": {"fusion": "rrf"},
            "limit": limit,
            "with_payload": True,
        }

    async def search_hybrid(
        self,
        embedding: HybridEmbedding,
        limit: int = 20,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = self.build_hybrid_query(embedding, limit)
        if document_id:
            payload["filter"] = {
                "must": [{"key": "document_id", "match": {"value": document_id}}]
            }
        response = await self.client.post(
            f"{self.base_url}/collections/{self.collection}/points/query",
            json=payload,
        )
        response.raise_for_status()
        result = response.json().get("result", {})
        return result.get("points", []) if isinstance(result, dict) else result

    async def delete_document(self, document_id: str) -> bool:
        response = await self.client.post(
            f"{self.base_url}/collections/{self.collection}/points/delete",
            params={"wait": "true"},
            json={
                "filter": {
                    "must": [{"key": "document_id", "match": {"value": document_id}}]
                }
            },
        )
        response.raise_for_status()
        return response.json().get("status") == "ok"

    async def close(self) -> None:
        await self.client.aclose()
