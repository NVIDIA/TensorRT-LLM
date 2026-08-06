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

import os
from typing import TypeAlias

from .cute_dsl_mla import CuteDslMlaFmha
from .fallback import FallbackFmha
from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from .interface import Fmha
from .prims_ts import PrimsTSFmha
from .triton_custom_mask import TritonCustomMaskFmha

FmhaCls: TypeAlias = type[Fmha]


def init_fmha_libs() -> dict[str, "FmhaCls"]:
    """Build the ordered FMHA library registry.

    Backend classes are imported inside this factory rather than at module
    scope, so backends can import trtllm attention classes at module scope
    without an import cycle.
    """
    from .flashinfer_sparse_mla import FlashInferSparseMlaFmha
    from .msa_sparse_gqa import MsaSparseGqaFmha

    return {
        "triton_custom_mask": TritonCustomMaskFmha,
        "prims_ts": PrimsTSFmha,
        "cute_dsl_mla": CuteDslMlaFmha,
        "msa_sparse_gqa": MsaSparseGqaFmha,
        "flashinfer_sparse_mla": FlashInferSparseMlaFmha,
        "flashinfer_trtllm_gen": FlashInferTrtllmGenFmha,
        "fallback": FallbackFmha,
    }


FMHA_LIBS: dict[str, FmhaCls] = init_fmha_libs()
DEFAULT_FMHA_LIBS: tuple[str, ...] = tuple(name for name in FMHA_LIBS if name != "prims_ts")


def _parse_fmha_libs_env() -> tuple[str, ...]:
    value = os.environ.get("TLLM_FMHA_LIBS")
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


__all__ = [
    "DEFAULT_FMHA_LIBS",
    "FMHA_LIBS",
    "FmhaCls",
    "get_enabled_fmha_lib_classes",
    "init_fmha_libs",
]
