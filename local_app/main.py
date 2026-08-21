import hashlib
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from common.local_storage import configure_local_storage


configure_local_storage()

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from api.routes.ask import AskRequest, AskResponse, Source, SYSTEM_PROMPT
from api.services.llm_client import LLMClient
from document_processor.chunker import DocumentChunker
from document_processor.cleaner import clean_text
from document_processor.main import decode_text, extract_text_from_pdf
from embedder_service.config import config as embedding_config
from embedder_service.model_loader import EmbeddingModel
from reranker_service import main as reranker

from .store import LocalVectorStore


ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA = Path(os.getenv("LOCAL_DATA_DIR", ROOT / ".local"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(512 * 1024 * 1024)))
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class Runtime:
    def __init__(self) -> None:
        self.embedder = EmbeddingModel(embedding_config)
        self.store: LocalVectorStore | None = None
        self.llm: LLMClient | None = None

    def load(self) -> None:
        self.embedder.load()
        if not self.embedder.loaded:
            raise RuntimeError(self.embedder.load_error or "BGE-M3 failed to load")
        reranker.load_model()
        if reranker.model is None:
            raise RuntimeError(reranker.load_error or "Qwen3 reranker failed to load")
        self.store = LocalVectorStore(LOCAL_DATA / "qdrant")
        self.llm = LLMClient(
            base_url=os.getenv("LLAMA_URL", "http://127.0.0.1:8001"),
            model=os.getenv("LLM_MODEL", "Qwen3-8B-Q4_K_M.gguf"),
        )

    def rerank(self, query: str, documents: list[str], top_k: int) -> list[dict[str, Any]]:
        scores: list[float] = []
        for start in range(0, len(documents), reranker.BATCH_SIZE):
            scores.extend(
                reranker.score_batch(
                    query,
                    documents[start : start + reranker.BATCH_SIZE],
                    reranker.DEFAULT_INSTRUCTION,
                )
            )
        ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
        return [
            {"text": document, "score": score}
            for document, score in ranked[:top_k]
        ]

    async def close(self) -> None:
        if self.store is not None:
            self.store.close()
        if self.llm is not None:
            await self.llm.close()


runtime = Runtime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_in_threadpool(runtime.load)
    yield
    await runtime.close()


app = FastAPI(
    title="Local RAG — native Windows mode",
    version="0.3.0",
    lifespan=lifespan,
)


def require_store() -> LocalVectorStore:
    if runtime.store is None:
        raise HTTPException(status_code=503, detail="Vector store is not initialized")
    return runtime.store


def process_document_bytes(
    content: bytes,
    filename: str,
    chunk_size: int,
    overlap: int,
    strategy: str,
) -> dict[str, Any]:
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        text, pages, method = extract_text_from_pdf(content)
    else:
        text, pages, method = clean_text(decode_text(content)), 1, "plain_text"
    if len(text.strip()) < 10:
        raise ValueError("No readable text was extracted")
    chunks = DocumentChunker(chunk_size=chunk_size, overlap=overlap).chunk_text(text, strategy)
    return {
        "chunks_data": chunks,
        "pages": pages,
        "method": method,
        "total_chars": len(text),
    }


@app.get("/health", include_in_schema=False)
@app.get("/api/v1/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "mode": "native-windows",
        "embedding_model": embedding_config.model_name,
        "reranker_model": reranker.MODEL_NAME,
        "reranker_device": reranker.device,
        "llm_backend": "llama.cpp",
    }


@app.post("/api/v1/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000, ge=200, le=4000),
    overlap: int = Form(150, ge=0, le=1000),
    strategy: str = Form("recursive", pattern="^(recursive|sentence|paragraph)$"),
) -> dict[str, Any]:
    filename = Path(file.filename or "unknown").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported formats: PDF, TXT and MD")
    if overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="overlap must be smaller than chunk_size")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        max_upload_mib = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Файл превышает допустимый размер {max_upload_mib} МБ",
        )

    store = require_store()
    content_hash = hashlib.sha256(content).hexdigest()
    duplicate = await run_in_threadpool(store.find_document_by_hash, content_hash)
    if duplicate:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "This document is already indexed",
                "document_id": duplicate.get("document_id"),
                "filename": duplicate.get("filename"),
            },
        )

    try:
        processed = await run_in_threadpool(
            process_document_bytes,
            content,
            filename,
            chunk_size,
            overlap,
            strategy,
        )
        chunks = processed["chunks_data"]
        vectors = await run_in_threadpool(
            runtime.embedder.encode,
            [chunk["text"] for chunk in chunks],
        )
        document_id = str(uuid.uuid4())
        uploaded_at = datetime.now(UTC).isoformat()
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": {
                    "dense": vector.dense,
                    "sparse": {
                        "indices": vector.sparse_indices,
                        "values": vector.sparse_values,
                    },
                },
                "payload": {
                    "text": chunk["text"],
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                    "uploaded_at": uploaded_at,
                    "content_hash": content_hash,
                    "pages": processed["pages"],
                },
            }
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        await run_in_threadpool(store.upsert_points, points)
        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "pages": processed["pages"],
            "chunks_uploaded": len(chunks),
            "total_chars": processed["total_chars"],
            "method": processed["method"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Native ingestion failed")
        raise HTTPException(status_code=500, detail="Document ingestion failed") from exc


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    store = require_store()
    if not await run_in_threadpool(store.list_documents):
        return AskResponse(answer="Сначала загрузите хотя бы один документ.", sources=[])
    try:
        query_embedding = (await run_in_threadpool(runtime.embedder.encode, [request.question]))[0]
        from api.services.embedder_client import HybridEmbedding, SparseVector

        embedding = HybridEmbedding(
            dense=query_embedding.dense,
            sparse=SparseVector(
                indices=query_embedding.sparse_indices,
                values=query_embedding.sparse_values,
            ),
        )
        candidates = await run_in_threadpool(
            store.search_hybrid,
            embedding,
            request.search_limit,
            request.document_id,
        )
        candidates = [point for point in candidates if (point.payload or {}).get("text", "").strip()]
        if not candidates:
            return AskResponse(
                answer="В загруженных документах не найдено информации по вашему вопросу.",
                sources=[],
            )

        documents = [point.payload["text"] for point in candidates]
        reranked = await run_in_threadpool(runtime.rerank, request.question, documents, request.top_k)
        payload_by_text = {point.payload["text"]: point.payload for point in candidates}
        sources: list[Source] = []
        context_parts: list[str] = []
        for result in reranked:
            payload = payload_by_text.get(result["text"])
            if payload is None:
                continue
            context_parts.append(result["text"])
            sources.append(
                Source(
                    text=result["text"],
                    score=result["score"],
                    document_id=payload.get("document_id", "unknown"),
                    filename=payload.get("filename", "unknown"),
                )
            )
        if not context_parts:
            return AskResponse(
                answer="В загруженных документах не найдено информации по вашему вопросу.",
                sources=[],
            )
        if runtime.llm is None:
            raise RuntimeError("LLM client is not initialized")
        prompt = (
            "/no_think\nКОНТЕКСТ:\n"
            + "\n\n---\n\n".join(context_parts)
            + f"\n\nВОПРОС: {request.question}"
        )
        answer = await runtime.llm.generate(
            prompt,
            system_prompt=SYSTEM_PROMPT,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        return AskResponse(answer=answer, sources=sources)
    except Exception as exc:
        logger.exception("Native question pipeline failed")
        raise HTTPException(status_code=500, detail="Question pipeline failed") from exc


@app.get("/api/v1/documents")
async def list_documents() -> dict[str, list[dict[str, Any]]]:
    return {"documents": await run_in_threadpool(require_store().list_documents)}


@app.delete("/api/v1/documents/{document_id}")
async def delete_document(document_id: str) -> dict[str, str]:
    try:
        uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid document_id") from exc
    await run_in_threadpool(require_store().delete_document, document_id)
    return {"status": "deleted", "document_id": document_id}


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
