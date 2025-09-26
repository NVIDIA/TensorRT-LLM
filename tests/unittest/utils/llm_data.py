import os
from pathlib import Path
from typing import Optional


def llm_models_root(check=False) -> Optional[Path]:
    root = Path("/home/scratch.trt_llm_data/llm-models/")

    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ.get("LLM_MODELS_ROOT"))

    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")

    if check:
        assert root.exists(), \
        "You shall set LLM_MODELS_ROOT env or be able to access /home/scratch.trt_llm_data to run this test"

    return root if root.exists() else None


def llm_datasets_root() -> str:
    return os.path.join(llm_models_root(check=True), "datasets")
