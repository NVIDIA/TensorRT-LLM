# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vllm.config stand-in: the names the decode runtime imports."""
import enum
from types import SimpleNamespace


class CUDAGraphMode(enum.Enum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_DECODE_ONLY = 3
    FULL_AND_PIECEWISE = 4


class CompilationMode(enum.Enum):
    NONE = 0
    VLLM_COMPILE = 3


class SpeculativeConfig:  # imported, not used on the replica path
    pass


_CONFIG = SimpleNamespace(parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1))


def get_current_vllm_config():
    return _CONFIG
