import os
from typing import Any

import httpx


class DocumentProcessorClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0):
        self.base_url = (
            base_url or os.getenv("DOCUMENT_PROCESSOR_URL", "http://document_processor:8080")
        ).rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        chunk_size: int = 1000,
        overlap: int = 150,
        strategy: str = "recursive",
    ) -> dict[str, Any]:
        response = await self.client.post(
            f"{self.base_url}/process",
            files={"file": (filename, file_content, "application/octet-stream")},
            data={
                "chunk_size": str(chunk_size),
                "overlap": str(overlap),
                "strategy": strategy,
            },
        )
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        await self.client.aclose()
