import pytest

from embedder_service.config import Config
from embedder_service.model_loader import EmbeddingModel


pytestmark = pytest.mark.unit


class FakeBGEM3Model:
    def encode(self, *_args, **_kwargs):
        return {
            "dense_vecs": [[0.1] * 1024],
            "lexical_weights": [{"42": 0.75, "7": 0.25}],
        }


def test_converts_bge_output_to_qdrant_sparse_format() -> None:
    embedding_model = EmbeddingModel(Config())
    embedding_model.model = FakeBGEM3Model()
    vector = embedding_model.encode(["test"])[0]
    assert len(vector.dense) == 1024
    assert vector.sparse_indices == [7, 42]
    assert vector.sparse_values == [0.25, 0.75]
