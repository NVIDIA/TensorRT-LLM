# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import functools
import os
from typing import TypeAlias

from .cute_dsl_mla import CuteDslMlaFmha
from .fallback import FallbackFmha
from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from .fp4_mla import Fp4MlaFmha
from .interface import Fmha
from .prims_ts import PrimsTSFmha
from .sampling_shared_prefix import SamplingSharedPrefixFmha
from .triton_custom_mask import TritonCustomMaskFmha

FmhaCls: TypeAlias = type[Fmha]


def init_fmha_libs() -> dict[str, "FmhaCls"]:
    """Build the ordered FMHA library registry.

    Backend classes are imported inside this factory rather than at module
    scope, so backends can import trtllm attention classes at module scope
    without an import cycle.
    """
    from .flashinfer_sparse_mla import FlashInferSparseMlaFmha
    from .msa_decode import MsaDecodeFmha
    from .msa_prefill import MsaPrefillFmha
    from .prims_ts_block_sparse import PrimsTSBlockSparseFmha
    from .prims_ts_sparse_mla import PrimsTSSparseMlaFmha

    return {
        "fp4_mla": Fp4MlaFmha,
        "triton_custom_mask": TritonCustomMaskFmha,
        "cute_dsl_mla": CuteDslMlaFmha,
        # A pair, not a priority: msa_decode serves a MiniMax-M3 MSA layer's
        # generation phase and msa_prefill its context phase, and neither will
        # take the other's.
        "msa_decode": MsaDecodeFmha,
        "msa_prefill": MsaPrefillFmha,
        "flashinfer_sparse_mla": FlashInferSparseMlaFmha,
        "prims_ts": PrimsTSFmha,
        "prims_ts_block_sparse": PrimsTSBlockSparseFmha,
        # DeepSeek-V4 generation on the vendored PrimTS native sparse MLA
        # kernel. Enabled by default: it only reports itself available for
        # DeepSeek-V4 on SM100, so every other model and architecture never
        # sees it; everything else keeps the TRTLLM-Gen libraries below.
        "prims_ts_sparse_mla": PrimsTSSparseMlaFmha,
        # Opt-in shared-prefix generation path (TLLM_FMHA_LIBS=+sampling_shared_prefix).
        # It must precede flashinfer_trtllm_gen, which takes every generation
        # batch it supports. It declines eager batches without a shared prefix
        # but serves every n-sample batch a CUDA graph is captured for.
        "sampling_shared_prefix": SamplingSharedPrefixFmha,
        "flashinfer_trtllm_gen": FlashInferTrtllmGenFmha,
        "fallback": FallbackFmha,
    }


FMHA_LIBS: dict[str, FmhaCls] = init_fmha_libs()
# Libraries left out of the default set and enabled explicitly through
# TLLM_FMHA_LIBS: dense PrimTS can add host overhead; sampling_shared_prefix is a
# narrow-scope n-sample path (KV cache manager V2).
_OPT_IN_FMHA_LIBS: frozenset[str] = frozenset({"prims_ts", "sampling_shared_prefix"})
DEFAULT_FMHA_LIBS: tuple[str, ...] = tuple(
    name for name in FMHA_LIBS if name not in _OPT_IN_FMHA_LIBS
)


def _parse_fmha_libs_env() -> tuple[str, ...]:
    return _parse_fmha_libs_value(os.environ.get("TLLM_FMHA_LIBS"))


@functools.lru_cache(maxsize=8)
def _parse_fmha_libs_value(value: str | None) -> tuple[str, ...]:
    """Parse a ``TLLM_FMHA_LIBS`` value; cached because per-step host code queries it."""
    if value is None or not value.strip():
        return DEFAULT_FMHA_LIBS

    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        return DEFAULT_FMHA_LIBS

    has_delta_token = any(token[0] in "+-" for token in tokens)
    if has_delta_token and not all(token[0] in "+-" for token in tokens):
        raise ValueError(
            "TLLM_FMHA_LIBS must use either an exact comma-separated list "
            "or only +/- delta entries."
        )

    if has_delta_token:
        enabled_names = set(DEFAULT_FMHA_LIBS)
        for token in tokens:
            sign = token[0]
            name = token[1:].strip()
            if not name:
                raise ValueError(f"Invalid empty FMHA library entry in {value!r}.")
            if name not in FMHA_LIBS:
                raise ValueError(f"Unknown FMHA library {name!r} in TLLM_FMHA_LIBS.")
            if sign == "+":
                enabled_names.add(name)
            else:
                enabled_names.discard(name)
        names = [name for name in FMHA_LIBS if name in enabled_names]
    else:
        names = []
        for name in tokens:
            if name not in FMHA_LIBS:
                raise ValueError(f"Unknown FMHA library {name!r} in TLLM_FMHA_LIBS.")
            if name not in names:
                names.append(name)

    return tuple(names)


def get_enabled_fmha_lib_classes() -> list[FmhaCls]:
    return [FMHA_LIBS[name] for name in _parse_fmha_libs_env()]


def is_fmha_lib_enabled(name: str) -> bool:
    """Whether ``TLLM_FMHA_LIBS`` (or the default set) enables ``name``."""
    return name in _parse_fmha_libs_env()


__all__ = [
    "DEFAULT_FMHA_LIBS",
    "FMHA_LIBS",
    "FmhaCls",
    "get_enabled_fmha_lib_classes",
    "init_fmha_libs",
    "is_fmha_lib_enabled",
]
