import os

import pytest

from api.services.llm_client import LLMClient


pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1"),
]


async def test_llm_generates_text() -> None:
    client = LLMClient(base_url="http://localhost:8001")
    try:
        answer = await client.generate("Ответь одним словом: два плюс два?", max_tokens=16)
        assert answer.strip()
    finally:
        await client.close()
