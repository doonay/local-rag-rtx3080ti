import logging
import os
import uuid
import hashlib
from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..services.document_processor_client import DocumentProcessorClient
from ..services.embedder_client import EmbedderClient
from ..services.qdrant_client import QdrantClient


router = APIRouter()
logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000, ge=200, le=4000),
    overlap: int = Form(150, ge=0, le=1000),
    strategy: str = Form("recursive", pattern="^(recursive|sentence|paragraph)$"),
) -> dict:
    filename = Path(file.filename or "unknown").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Supported formats: PDF, TXT and MD")
    if overlap >= chunk_size:
        raise HTTPException(status_code=422, detail="overlap must be smaller than chunk_size")

    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="The uploaded file is too large")

    processor = DocumentProcessorClient()
    embedder = EmbedderClient()
    qdrant = QdrantClient()
    try:
        content_hash = hashlib.sha256(content).hexdigest()
        duplicate = await qdrant.find_document_by_hash(content_hash)
        if duplicate:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "This document is already indexed",
                    "document_id": duplicate.get("document_id"),
                    "filename": duplicate.get("filename"),
                },
            )
        processed = await processor.process_document(
            file_content=content,
            filename=filename,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
        )
        chunks = processed.get("chunks_data", [])
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text was extracted")

        embeddings = await embedder.embed([chunk["text"] for chunk in chunks])
        document_id = str(uuid.uuid4())
        uploaded_at = datetime.now(UTC).isoformat()
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": {
                    QdrantClient.DENSE_VECTOR_NAME: embedding.dense,
                    QdrantClient.SPARSE_VECTOR_NAME: {
                        "indices": embedding.sparse.indices,
                        "values": embedding.sparse.values,
                    },
                },
                "payload": {
                    "text": chunk["text"],
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                    "chunk_length": chunk.get("length", len(chunk["text"])),
                    "chunk_type": chunk.get("type", strategy),
                    "uploaded_at": uploaded_at,
                    "content_hash": content_hash,
                    "pages": processed.get("pages", 1),
                },
            }
            for index, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        if not await qdrant.upsert_points(points):
            raise RuntimeError("Qdrant did not confirm the upsert operation")
        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "pages": processed.get("pages", 1),
            "chunks_uploaded": len(chunks),
            "total_chars": processed.get("total_chars", 0),
            "method": processed.get("method", "unknown"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(status_code=502, detail="A RAG service failed during ingestion") from exc
    finally:
        await processor.close()
        await embedder.close()
        await qdrant.close()


@router.get("/documents")
async def list_documents() -> dict[str, list[dict]]:
    qdrant = QdrantClient()
    try:
        return {"documents": await qdrant.list_documents()}
    except Exception as exc:
        logger.exception("Could not list documents")
        raise HTTPException(status_code=502, detail="Qdrant is unavailable") from exc
    finally:
        await qdrant.close()


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> dict[str, str]:
    try:
        uuid.UUID(document_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid document_id") from exc

    qdrant = QdrantClient()
    try:
        if not await qdrant.delete_document(document_id):
            raise HTTPException(status_code=502, detail="Qdrant did not confirm deletion")
        return {"status": "deleted", "document_id": document_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not delete document")
        raise HTTPException(status_code=502, detail="Qdrant is unavailable") from exc
    finally:
        await qdrant.close()
