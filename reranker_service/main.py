import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from common.local_storage import configure_local_storage


configure_local_storage()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
USE_FP16 = os.getenv("RERANKER_USE_FP16", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "4"))
MAX_LENGTH = int(os.getenv("RERANKER_MAX_LENGTH", "2048"))
DEFAULT_INSTRUCTION = (
    "Given a search query, retrieve relevant passages that answer the query. "
    "Prefer passages containing a direct, factual answer."
)

model: Any | None = None
tokenizer: Any | None = None
torch_module: Any | None = None
device = "cpu"
load_error: str | None = None
token_false_id: int | None = None
token_true_id: int | None = None
prefix_tokens: list[int] = []
suffix_tokens: list[int] = []


def load_model() -> None:
    global model, tokenizer, torch_module, device, load_error
    global token_false_id, token_true_id, prefix_tokens, suffix_tokens
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_module = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" and USE_FP16 else torch.float32
        logger.info("Loading instruction-aware reranker %s on %s", MODEL_NAME, device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()

        token_false_id = tokenizer.convert_tokens_to_ids("no")
        token_true_id = tokenizer.convert_tokens_to_ids("yes")
        if not isinstance(token_false_id, int) or not isinstance(token_true_id, int):
            raise RuntimeError("Could not resolve Qwen3 reranker yes/no token IDs")
        prefix = (
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            "based on the Query and the Instruct provided. Note that the answer can only "
            "be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        load_error = None
        logger.info("Reranker loaded")
    except Exception as exc:
        model = tokenizer = torch_module = None
        load_error = str(exc)
        logger.exception("Reranker failed to load")


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    yield


app = FastAPI(title="Qwen3 Reranker Service", version="0.3.0", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    documents: list[str] = Field(min_length=1, max_length=100)
    top_k: int | None = Field(default=None, ge=1, le=100)
    instruction: str | None = Field(default=None, max_length=1000)


class RerankResult(BaseModel):
    text: str
    score: float
    rank: int


class RerankResponse(BaseModel):
    query: str
    results: list[RerankResult]
    total_documents: int


def format_instruction(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"


def score_batch(query: str, documents: list[str], instruction: str) -> list[float]:
    if model is None or tokenizer is None or torch_module is None:
        raise RuntimeError("Reranker model is unavailable")
    pairs = [format_instruction(instruction, query, document) for document in documents]
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=MAX_LENGTH - len(prefix_tokens) - len(suffix_tokens),
    )
    for index, input_ids in enumerate(inputs["input_ids"]):
        inputs["input_ids"][index] = prefix_tokens + input_ids + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=MAX_LENGTH)
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch_module.inference_mode():
        logits = model(**inputs).logits[:, -1, :]
        true_scores = logits[:, token_true_id]
        false_scores = logits[:, token_false_id]
        binary_scores = torch_module.stack([false_scores, true_scores], dim=1)
        return torch_module.nn.functional.log_softmax(binary_scores, dim=1)[:, 1].exp().tolist()


@app.get("/health")
async def health() -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail=load_error or "Reranker is not loaded")
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "model_loaded": True,
        "device": device,
        "use_fp16": USE_FP16 and device == "cuda",
        "max_length": MAX_LENGTH,
    }


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Reranker model is unavailable")
    documents = [document.strip() for document in request.documents]
    if any(not document for document in documents):
        raise HTTPException(status_code=422, detail="Documents must be non-empty")

    instruction = request.instruction or DEFAULT_INSTRUCTION
    scores: list[float] = []
    try:
        for start in range(0, len(documents), BATCH_SIZE):
            scores.extend(score_batch(request.query, documents[start : start + BATCH_SIZE], instruction))
    except Exception as exc:
        logger.exception("Reranking failed")
        raise HTTPException(status_code=500, detail="Reranking failed") from exc

    ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
    if request.top_k is not None:
        ranked = ranked[: request.top_k]
    results = [
        RerankResult(text=document, score=score, rank=rank)
        for rank, (document, score) in enumerate(ranked, start=1)
    ]
    return RerankResponse(query=request.query, results=results, total_documents=len(documents))


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "Qwen3 Reranker Service", "version": "0.3.0"}
