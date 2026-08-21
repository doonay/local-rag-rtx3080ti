import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import config
from .model_loader import EmbeddingModel


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
model = EmbeddingModel(config)


@asynccontextmanager
async def lifespan(_: FastAPI):
    model.load()
    yield


app = FastAPI(title="BGE-M3 Hybrid Embedding Service", version="0.3.0", lifespan=lifespan)


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=256)


class SparseEmbedding(BaseModel):
    indices: list[int]
    values: list[float]


class HybridEmbedding(BaseModel):
    dense: list[float]
    sparse: SparseEmbedding


class EmbedResponse(BaseModel):
    vectors: list[HybridEmbedding]
    dimension: int
    count: int


@app.get("/health")
async def health() -> dict:
    if not model.loaded:
        raise HTTPException(status_code=503, detail=model.load_error or "Embedding model is not loaded")
    return {
        "status": "healthy",
        "model": config.model_name,
        "model_loaded": True,
        "dimension": model.dimension,
        "device": config.device,
        "modes": ["dense", "sparse"],
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse:
    texts = [text.strip() for text in request.texts]
    if any(not text for text in texts):
        raise HTTPException(status_code=422, detail="Texts must be non-empty")
    if not model.loaded:
        raise HTTPException(status_code=503, detail="Embedding model is unavailable")
    try:
        encoded = model.encode(texts)
        vectors = [
            HybridEmbedding(
                dense=vector.dense,
                sparse=SparseEmbedding(
                    indices=vector.sparse_indices,
                    values=vector.sparse_values,
                ),
            )
            for vector in encoded
        ]
        return EmbedResponse(vectors=vectors, dimension=model.dimension, count=len(vectors))
    except Exception as exc:
        logging.getLogger(__name__).exception("Embedding failed")
        raise HTTPException(status_code=500, detail="Embedding failed") from exc


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "BGE-M3 Hybrid Embedding Service", "version": "0.3.0"}
