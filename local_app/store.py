from pathlib import Path
from typing import Any, Sequence

from qdrant_client import QdrantClient, models

from .embeddings import HybridEmbedding


class LocalVectorStore:
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self, path: Path, collection: str = "documents_local"):
        path.mkdir(parents=True, exist_ok=True)
        self.collection = collection
        self.client = QdrantClient(path=str(path))

    def ensure_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection):
            collection = self.client.get_collection(self.collection)
            vectors = collection.config.params.vectors
            dense = vectors.get(self.DENSE_VECTOR_NAME) if isinstance(vectors, dict) else None
            if dense is None or dense.size != vector_size:
                raise RuntimeError(
                    "Local Qdrant index has an incompatible vector schema. "
                    "Remove .local/qdrant and re-index the documents."
                )
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                self.DENSE_VECTOR_NAME: models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                self.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )

    def find_document_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        if not self.client.collection_exists(self.collection):
            return None
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="content_hash",
                        match=models.MatchValue(value=content_hash),
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return dict(points[0].payload or {}) if points else None

    def upsert_points(self, points: Sequence[dict[str, Any]]) -> None:
        if not points:
            raise ValueError("No points supplied")
        dense = points[0]["vector"][self.DENSE_VECTOR_NAME]
        self.ensure_collection(len(dense))
        self.client.upsert(
            collection_name=self.collection,
            wait=True,
            points=[
                models.PointStruct(
                    id=point["id"],
                    vector={
                        self.DENSE_VECTOR_NAME: point["vector"][self.DENSE_VECTOR_NAME],
                        self.SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=point["vector"][self.SPARSE_VECTOR_NAME]["indices"],
                            values=point["vector"][self.SPARSE_VECTOR_NAME]["values"],
                        ),
                    },
                    payload=point["payload"],
                )
                for point in points
            ],
        )

    def search_hybrid(
        self,
        embedding: HybridEmbedding,
        limit: int = 20,
        document_id: str | None = None,
    ) -> list[Any]:
        query_filter = None
        if document_id:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            )
        if not embedding.sparse.indices:
            response = self.client.query_points(
                collection_name=self.collection,
                query=embedding.dense,
                using=self.DENSE_VECTOR_NAME,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
            return response.points

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                models.Prefetch(
                    query=embedding.dense,
                    using=self.DENSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.SparseVector(
                        indices=embedding.sparse.indices,
                        values=embedding.sparse.values,
                    ),
                    using=self.SPARSE_VECTOR_NAME,
                    filter=query_filter,
                    limit=limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return response.points

    def list_documents(self) -> list[dict[str, Any]]:
        if not self.client.collection_exists(self.collection):
            return []
        documents: dict[str, dict[str, Any]] = {}
        offset: Any | None = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = dict(point.payload or {})
                document_id = payload.get("document_id")
                if document_id and document_id not in documents:
                    documents[document_id] = {
                        "document_id": document_id,
                        "filename": payload.get("filename", "unknown"),
                        "chunks": payload.get("total_chunks", 0),
                        "pages": payload.get("pages", 1),
                        "uploaded_at": payload.get("uploaded_at"),
                        "content_hash": payload.get("content_hash"),
                    }
            if offset is None:
                break
        return sorted(
            documents.values(),
            key=lambda document: document.get("uploaded_at") or "",
            reverse=True,
        )

    def delete_document(self, document_id: str) -> None:
        if not self.client.collection_exists(self.collection):
            return
        self.client.delete(
            collection_name=self.collection,
            wait=True,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def close(self) -> None:
        self.client.close()
