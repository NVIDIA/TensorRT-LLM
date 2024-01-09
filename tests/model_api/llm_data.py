import os
from pathlib import Path


def llm_models_root() -> str:
    root = Path(
        os.environ.get('LLM_MODELS_ROOT',
                       '/home/scratch.trt_llm_data/llm-models/'))
    assert root.exists(), \
    "You shall set LLM_MODELS_ROOT env or be able to access /home/scratch.trt_llm_data to run this test"
    return root
