import os
from typing import Any, Sequence

import httpx


class RerankerClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or os.getenv("RERANKER_URL", "http://reranker:8080")).rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []
        payload: dict[str, Any] = {"query": query, "documents": list(documents)}
        if top_k is not None:
            payload["top_k"] = top_k
        response = await self.client.post(f"{self.base_url}/rerank", json=payload)
        response.raise_for_status()
        return response.json().get("results", [])

    async def health_check(self) -> dict:
        response = await self.client.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()
