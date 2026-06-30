# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compatibility layer for running auto_deploy with or without TensorRT-LLM.

This module is the SINGLE place that probes for TensorRT-LLM availability.
All other auto_deploy modules should import TRT-LLM-provided types and utilities
from here rather than directly from tensorrt_llm.

When TRT-LLM IS available, the real implementations are re-exported (preserving
type identity). When TRT-LLM is NOT available, lightweight standalone replacements
are provided.

Only types with genuine standalone consumers are reimplemented here. Types used
exclusively by TRT-LLM-only files (which get skipped entirely via import guards)
are NOT duplicated.
"""

import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Probe for TensorRT-LLM
# ---------------------------------------------------------------------------
# We probe for a submodule that only exists in the real TRT-LLM package.
# A stub tensorrt_llm package (used in CI standalone tests) won't have this,
# so the probe correctly returns False in that environment.
try:
    from tensorrt_llm import mapping as _  # noqa: F401

    TRTLLM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    TRTLLM_AVAILABLE = False

# ---------------------------------------------------------------------------
# ModelOpt quant config helpers  (used by standalone model factories)
# ---------------------------------------------------------------------------
if TRTLLM_AVAILABLE:
    from tensorrt_llm.quantization.modelopt_config import (
        is_modelopt_quant_config,
        read_modelopt_quant_config,
    )
else:

    def is_modelopt_quant_config(raw: Any) -> bool:
        """Return True if ``raw`` looks like a ModelOpt quantization config."""
        if not isinstance(raw, dict):
            return False
        if str(raw.get("quant_method", "")).lower().startswith("modelopt"):
            return True
        return (raw.get("producer") or {}).get("name") == "modelopt"

    _KV_SCHEME_DICT_MAP = {
        ("float", 8): "FP8",
        ("float", 4): "NVFP4",
        ("int", 8): "INT8",
    }
    _KV_SCHEME_STRING_ALGOS = {"FP8", "NVFP4", "INT8"}

    def _kv_cache_scheme_to_algo(scheme: Any) -> Optional[str]:
        if scheme is None:
            return None
        if isinstance(scheme, str):
            algo = scheme.upper()
            if algo in _KV_SCHEME_STRING_ALGOS:
                return algo
        elif isinstance(scheme, dict):
            return _KV_SCHEME_DICT_MAP.get((scheme.get("type"), scheme.get("num_bits")))
        return None

    def read_modelopt_quant_config(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize supported ModelOpt quant config shapes for standalone use."""
        if not isinstance(raw, dict):
            raise ValueError(f"Expected dict for modelopt quant config, got {type(raw).__name__}")
        if "quantization" in raw:
            quantization = raw["quantization"]
            if not isinstance(quantization, dict):
                raise ValueError(
                    f"'quantization' must be a dict, got {type(quantization).__name__}"
                )
            result = dict(quantization)
        elif is_modelopt_quant_config(raw):
            skip_keys = {"producer", "quant_method", "ignore", "kv_cache_scheme", "config_groups"}
            result = {key: value for key, value in raw.items() if key not in skip_keys}
            if "ignore" in raw:
                result["exclude_modules"] = raw["ignore"]
            if "kv_cache_scheme" in raw:
                algo = _kv_cache_scheme_to_algo(raw["kv_cache_scheme"])
                if algo is not None:
                    result["kv_cache_quant_algo"] = algo
        else:
            raise ValueError(
                f"Not a modelopt quant config (producer={raw.get('producer')!r}, "
                f"quant_method={raw.get('quant_method')!r})"
            )
        if result.get("quant_algo") == "fp8_pb_wo":
            result["quant_algo"] = "FP8_BLOCK_SCALES"
        return result


# ---------------------------------------------------------------------------
# ActivationType  (used by 11 standalone files)
# ---------------------------------------------------------------------------
if TRTLLM_AVAILABLE:
    from tensorrt_llm._torch.utils import ActivationType
else:

    class ActivationType(IntEnum):
        InvalidType = 0
        Identity = 1
        Gelu = 2
        Relu = 3
        Silu = 4
        Swiglu = 5
        Geglu = 6
        SwigluBias = 7
        Relu2 = 8


# ---------------------------------------------------------------------------
# MultimodalInput  (used by standalone multimodal input processors)
# ---------------------------------------------------------------------------
if TRTLLM_AVAILABLE:
    from tensorrt_llm.inputs.multimodal import MultimodalInput
else:

    @dataclass
    class MultimodalInput:
        """Standalone subset of TensorRT-LLM's multimodal request metadata."""

        multimodal_hashes: List[List[int]]
        multimodal_positions: List[int]
        multimodal_lengths: List[int]

        @classmethod
        def from_components(
            cls,
            mm_hashes: List[List[int]],
            mm_positions: List[int],
            mm_lengths: List[int],
            **_: object,
        ) -> "MultimodalInput":
            return cls(
                multimodal_hashes=mm_hashes,
                multimodal_positions=mm_positions,
                multimodal_lengths=mm_lengths,
            )


# ---------------------------------------------------------------------------
# AllReduceStrategy  (used by sharding.py which is standalone)
# ---------------------------------------------------------------------------
if TRTLLM_AVAILABLE:
    from tensorrt_llm.functional import AllReduceStrategy
else:

    class AllReduceStrategy(IntEnum):
        NCCL = 0
        MIN_LATENCY = 1
        UB = 2
        AUTO = 3
        ONESHOT = 4
        TWOSHOT = 5
        LOWPRECISION = 6
        MNNVL = 7
        NCCL_SYMMETRIC = 8
        SYMM_MEM = 9


# ---------------------------------------------------------------------------
# KvCacheConfig  (used by 16 standalone files)
# ---------------------------------------------------------------------------
if TRTLLM_AVAILABLE:
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
else:
    from pydantic import BaseModel

    class KvCacheConfig(BaseModel):
        """Standalone stub for KvCacheConfig with fields auto_deploy reads."""

        enable_block_reuse: bool = True
        max_tokens: Optional[int] = None
        max_attention_window: Optional[List[int]] = None
        sink_token_length: Optional[int] = None
        free_gpu_memory_fraction: Optional[float] = 0.9
        host_cache_size: Optional[int] = None
        onboard_blocks: bool = True
        cross_kv_cache_fraction: Optional[float] = None
        secondary_offload_min_priority: Optional[int] = None
        event_buffer_max_size: int = 0
        kv_cache_event_hash_algo: str = "auto"
        enable_partial_reuse: bool = True
        copy_on_partial_reuse: bool = True
        use_uvm: bool = False
        tokens_per_block: int = 32
        dtype: str = "auto"

        model_config = {"arbitrary_types_allowed": True, "extra": "allow"}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
if TRTLLM_AVAILABLE:
    from tensorrt_llm._torch.utils import make_weak_ref
    from tensorrt_llm._utils import (
        get_free_port,
        get_sm_version,
        is_sm_100f,
        mpi_disabled,
        nvtx_range,
        prefer_pinned,
        str_dtype_to_torch,
    )
else:
    # -- get_free_port --
    def get_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    # -- get_sm_version --
    @lru_cache(maxsize=1)
    def get_sm_version() -> int:
        prop = torch.cuda.get_device_properties(0)
        return prop.major * 10 + prop.minor

    # -- is_sm_100f --
    @lru_cache(maxsize=1)
    def is_sm_100f(sm_version=None) -> bool:
        if sm_version is None:
            sm_version = get_sm_version()
        return sm_version == 100 or sm_version == 103

    # -- nvtx_range --
    @contextmanager
    def nvtx_range(msg, color="grey", domain="auto_deploy", category=None):
        try:
            import nvtx

            with nvtx.annotate(msg, color=color, domain=domain, category=category):
                yield
        except ImportError:
            yield

    # -- prefer_pinned --
    @lru_cache(maxsize=1)
    def prefer_pinned() -> bool:
        return True  # safe default in standalone mode

    # -- str_dtype_to_torch --
    _STR_TO_TORCH_DTYPE = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "int64": torch.int64,
        "int32": torch.int32,
        "int8": torch.int8,
        "bool": torch.bool,
        "fp8": torch.float8_e4m3fn,
    }

    def str_dtype_to_torch(dtype: str) -> torch.dtype:
        ret = _STR_TO_TORCH_DTYPE.get(dtype)
        assert ret is not None, f"Unsupported dtype: {dtype}"
        return ret

    # -- mpi_disabled --
    def mpi_disabled() -> bool:
        return os.environ.get("TLLM_DISABLE_MPI") == "1"

    # -- make_weak_ref --
    def make_weak_ref(x):
        """Identity passthrough in standalone mode (no TensorWrapper)."""
        if isinstance(x, (torch.Tensor, int, float, bool)):
            return x
        elif isinstance(x, tuple):
            return tuple(make_weak_ref(i) for i in x)
        elif isinstance(x, list):
            return [make_weak_ref(i) for i in x]
        elif isinstance(x, dict):
            return {k: make_weak_ref(v) for k, v in x.items()}
        else:
            return x
