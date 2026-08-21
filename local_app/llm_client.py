import os

import httpx


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 240.0,
    ):
        self.base_url = (
            base_url or os.getenv("LLAMA_URL", "http://127.0.0.1:8001")
        ).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", "Qwen3-8B-Q4_K_M.gguf")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        response = await self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.8,
                "repetition_penalty": 1.05,
                "chat_template_kwargs": {"enable_thinking": False},
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    async def close(self) -> None:
        await self.client.aclose()
