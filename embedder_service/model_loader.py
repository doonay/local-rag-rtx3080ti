import logging
from dataclasses import dataclass
from typing import Any, Sequence

from common.local_storage import configure_local_storage

from .config import Config


configure_local_storage()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridVector:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class EmbeddingModel:
    def __init__(self, config: Config):
        self.config = config
        self.model: Any | None = None
        self.load_error: str | None = None

    def load(self) -> None:
        try:
            from FlagEmbedding import BGEM3FlagModel

            logger.info("Loading hybrid embedding model %s", self.config.model_name)
            self.model = BGEM3FlagModel(
                self.config.model_name,
                use_fp16=self.config.device == "cuda",
                devices=self.config.device,
                query_max_length=self.config.max_sequence_length,
                passage_max_length=self.config.max_sequence_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            self.load_error = None
            logger.info("Hybrid embedding model loaded; dense dimension=%s", self.dimension)
        except Exception as exc:
            self.model = None
            self.load_error = str(exc)
            logger.exception("Embedding model failed to load")

    @property
    def loaded(self) -> bool:
        return self.model is not None

    @property
    def dimension(self) -> int:
        return self.config.dimension if self.model is not None else 0

    def encode(self, texts: Sequence[str]) -> list[HybridVector]:
        if self.model is None:
            raise RuntimeError(self.load_error or "Embedding model is not loaded")
        output = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            max_length=self.config.max_sequence_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_vectors = output["dense_vecs"]
        sparse_vectors = output["lexical_weights"]
        vectors: list[HybridVector] = []
        for dense, sparse in zip(dense_vectors, sparse_vectors, strict=True):
            if len(dense) != self.config.dimension:
                raise RuntimeError(
                    f"Expected dense vectors of size {self.config.dimension}, got {len(dense)}"
                )
            sorted_items = sorted((int(index), float(value)) for index, value in sparse.items())
            vectors.append(
                HybridVector(
                    dense=[float(value) for value in dense],
                    sparse_indices=[item[0] for item in sorted_items],
                    sparse_values=[item[1] for item in sorted_items],
                )
            )
        return vectors
