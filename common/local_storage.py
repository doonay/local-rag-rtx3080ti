"""Keep runtime data and third-party caches inside this project."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_local_storage() -> Path:
    """Configure local paths before ML libraries resolve their cache constants."""
    local_root = Path(
        os.environ.get("RAG_PROJECT_LOCAL_ROOT", PROJECT_ROOT / ".local")
    ).resolve()
    cache_root = local_root / "cache"
    huggingface_root = cache_root / "huggingface"
    temp_root = local_root / "tmp"

    local_paths = {
        "RAG_PROJECT_LOCAL_ROOT": local_root,
        "LOCAL_DATA_DIR": local_root,
        "HF_HOME": huggingface_root,
        "HF_HUB_CACHE": huggingface_root / "hub",
        "HUGGINGFACE_HUB_CACHE": huggingface_root / "hub",
        "HF_XET_CACHE": huggingface_root / "xet",
        "HF_ASSETS_CACHE": huggingface_root / "assets",
        "HF_DATASETS_CACHE": huggingface_root / "datasets",
        "TRANSFORMERS_CACHE": huggingface_root / "hub",
        "SENTENCE_TRANSFORMERS_HOME": cache_root / "sentence-transformers",
        "TORCH_HOME": cache_root / "torch",
        "XDG_CACHE_HOME": cache_root,
        "XDG_CONFIG_HOME": local_root / "config",
        "XDG_DATA_HOME": local_root / "share",
        "XDG_STATE_HOME": local_root / "state",
        "PIP_CACHE_DIR": cache_root / "pip",
        "UV_CACHE_DIR": cache_root / "uv",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "NUMBA_CACHE_DIR": cache_root / "numba",
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "CUDA_CACHE_PATH": cache_root / "nvidia",
        "PYTHONPYCACHEPREFIX": cache_root / "python",
        "TEMP": temp_root,
        "TMP": temp_root,
        "TMPDIR": temp_root,
    }

    for name, path in local_paths.items():
        resolved_path = Path(path).resolve()
        os.environ[name] = str(resolved_path)
        resolved_path.mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = os.environ["PYTHONPYCACHEPREFIX"]
    return local_root
