import os
from dataclasses import dataclass
from typing import Sequence

import httpx


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class HybridEmbedding:
    dense: list[float]
    sparse: SparseVector


class EmbedderClient:
    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = (base_url or os.getenv("EMBEDDER_URL", "http://embedder:8080")).rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def embed(self, texts: Sequence[str]) -> list[HybridEmbedding]:
        cleaned = [text.strip() for text in texts if text and text.strip()]
        if len(cleaned) != len(texts):
            raise ValueError("Texts must be non-empty strings")
        if not cleaned:
            raise ValueError("No texts provided")

        response = await self.client.post(f"{self.base_url}/embed", json={"texts": cleaned})
        response.raise_for_status()
        raw_vectors = response.json().get("vectors", [])
        if len(raw_vectors) != len(cleaned):
            raise RuntimeError("Embedder returned an unexpected number of vectors")
        return [
            HybridEmbedding(
                dense=[float(value) for value in vector["dense"]],
                sparse=SparseVector(
                    indices=[int(index) for index in vector["sparse"]["indices"]],
                    values=[float(value) for value in vector["sparse"]["values"]],
                ),
            )
            for vector in raw_vectors
        ]

    async def embed_single(self, text: str) -> HybridEmbedding:
        return (await self.embed([text]))[0]

    async def health_check(self) -> dict:
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()
