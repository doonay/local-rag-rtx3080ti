import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.embedder_client import EmbedderClient
from ..services.llm_client import LLMClient
from ..services.qdrant_client import QdrantClient
from ..services.reranker_client import RerankerClient


router = APIRouter()
logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=10)
    search_limit: int = Field(default=20, ge=1, le=50)
    temperature: float = Field(default=0.1, ge=0.0, le=1.5)
    max_tokens: int = Field(default=512, ge=32, le=2048)


class Source(BaseModel):
    text: str
    score: float
    document_id: str
    filename: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


SYSTEM_PROMPT = """Ты отвечаешь только на основании предоставленного контекста.
Если контекст не содержит ответа, скажи: «В предоставленных документах нет информации по этому вопросу».
Не добавляй факты из собственных знаний. Отвечай на языке вопроса, кратко и по делу."""


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    embedder = EmbedderClient()
    qdrant = QdrantClient()
    reranker = RerankerClient()
    llm = LLMClient()
    try:
        query_embedding = await embedder.embed_single(request.question)
        search_results = await qdrant.search_hybrid(
            query_embedding,
            limit=request.search_limit,
            document_id=request.document_id,
        )
        candidates = [
            result for result in search_results if result.get("payload", {}).get("text", "").strip()
        ]
        if not candidates:
            return AskResponse(
                answer="В загруженных документах не найдено информации по вашему вопросу.",
                sources=[],
            )

        documents = [result["payload"]["text"] for result in candidates]
        reranked = await reranker.rerank(request.question, documents, request.top_k)
        source_by_text = {result["payload"]["text"]: result for result in candidates}
        sources: list[Source] = []
        context_parts: list[str] = []
        for result in reranked:
            text = result.get("text", "").strip()
            original = source_by_text.get(text)
            if not original:
                continue
            payload = original["payload"]
            context_parts.append(text)
            sources.append(
                Source(
                    text=text,
                    score=float(result.get("score", 0.0)),
                    document_id=payload.get("document_id", "unknown"),
                    filename=payload.get("filename", "unknown"),
                )
            )
        if not context_parts:
            return AskResponse(
                answer="В загруженных документах не найдено информации по вашему вопросу.",
                sources=[],
            )

        context = "\n\n---\n\n".join(context_parts)
        prompt = f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {request.question}"
        answer = await llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return AskResponse(answer=answer, sources=sources)
    except Exception as exc:
        logger.exception("Question pipeline failed")
        raise HTTPException(status_code=502, detail="A RAG service failed while answering") from exc
    finally:
        await embedder.close()
        await qdrant.close()
        await reranker.close()
        await llm.close()
