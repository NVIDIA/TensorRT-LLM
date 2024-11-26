from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, computed_field, model_validator

from tensorrt_llm.bench.utils import (VALID_CACHE_DTYPES, VALID_COMPUTE_DTYPES,
                                      VALID_QUANT_ALGOS)


class EngineConstraints(BaseModel):
    max_batch_size: int = 2048
    max_tokens: int = 2048
    max_sequence_length: int = 6144
    tp_size: int = 1
    pp_size: int = 1

    @computed_field
    def world_size(self) -> int:
        return self.tp_size * self.pp_size


class EngineConfiguration(BaseModel):
    quantization: Optional[VALID_QUANT_ALGOS] = None
    kv_cache_dtype: Optional[VALID_CACHE_DTYPES] = "float16"
    fused_mlp: Optional[bool] = False
    dtype: Optional[VALID_COMPUTE_DTYPES] = "float16"
    gemm_plugin: Optional[bool] = False
    gpt_attn_plugin: Optional[bool] = True
    paged_context_fmha: Optional[bool] = True
    gemm_swiglu_plugin: Optional[bool] = False
    multi_block_mode: Optional[bool] = False
    multiple_profiles: Optional[bool] = True
    build_options: List[str] = []


class BuildConfiguration(BaseModel):
    model: str
    workspace: Path
    engine_dir: Optional[Path] = None
    engine_config: EngineConfiguration
    engine_limits: EngineConstraints

    @computed_field
    def get_build_feature_args(self) -> List[str]:
        ...

    @model_validator(mode="after")
    def check_engine_dir(self) -> BuildConfiguration:
        if self.engine_dir is None:
            limits = self.engine_limits
            engine_name: str = (
                f"BS_{limits.max_batch_size}_sl_{limits.max_sequence_length}_"
                f"tp_{limits.tp_size}_pp_{limits.pp_size}")
            self.engine_dir = Path(
                self.workspace,
                self.model,
                engine_name,
            )

        return self
