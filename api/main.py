import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import ask, upload


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="Local RAG API",
    description="Document ingestion and retrieval-augmented generation API",
    version="0.2.0",
)

allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/v1", tags=["documents"])
app.include_router(ask.router, prefix="/api/v1", tags=["questions"])


@app.get("/health", include_in_schema=False)
@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "api"}


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {"service": "Local RAG API", "docs": "/docs"}
