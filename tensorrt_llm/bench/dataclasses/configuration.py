from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from tensorrt_llm.llmapi.llm_args import BaseLlmArgs


class RuntimeConfig(BaseModel):
    model_path: Optional[Path] = None
    engine_dir: Optional[Path] = None
    revision: Optional[str] = None
    sw_version: str
    backend: Literal["pytorch", "_autodeploy", None] = None
    iteration_log: Optional[Path] = None
    llm_args: BaseLlmArgs
