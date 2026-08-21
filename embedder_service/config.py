import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    model_name: str = os.getenv("EMBEDDING_MODEL", os.getenv("MODEL_NAME", "BAAI/bge-m3"))
    device: str = os.getenv("EMBEDDING_DEVICE", os.getenv("DEVICE", "cpu"))
    dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
    max_sequence_length: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "1024"))
    normalize_embeddings: bool = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"


config = Config()
