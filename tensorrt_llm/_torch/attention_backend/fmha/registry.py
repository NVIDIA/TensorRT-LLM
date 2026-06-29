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

from .fallback import FallbackFmha
from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from .interface import Fmha
from .msa_proxy_mqa import MsaProxyMqaFmha
from .msa_sparse_gqa import MsaSparseGqaFmha

FmhaCls: TypeAlias = type[Fmha]

FMHA_LIBS: dict[str, FmhaCls] = {
    "flashinfer_trtllm_gen": FlashInferTrtllmGenFmha,
    "fallback": FallbackFmha,
    # Indexer-style proxy FMHA. Returns False from is_supported() so the
    # main-attention dispatch loop ignores it; sparse-attention indexers
    # locate it via get_enabled_fmha_lib_classes() filtered to
    # subclasses of IndexerProxyFmha.
    "msa_proxy_mqa": MsaProxyMqaFmha,
    # Block-sparse main-attention FMHA. Consumes kv_block_indexes
    # produced by an upstream proxy + top-k pass. Same opt-out pattern
    # as the proxy: returns False from is_supported() and is invoked
    # directly by sparse-attention backends via forward_block_sparse().
    "msa_sparse_gqa": MsaSparseGqaFmha,
}
DEFAULT_FMHA_LIBS: tuple[str, ...] = tuple(FMHA_LIBS)


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
        names = list(DEFAULT_FMHA_LIBS)
        for token in tokens:
            sign = token[0]
            name = token[1:].strip()
            if not name:
                raise ValueError(f"Invalid empty FMHA library entry in {value!r}.")
            if name not in FMHA_LIBS:
                raise ValueError(f"Unknown FMHA library {name!r} in TLLM_FMHA_LIBS.")
            if sign == "+" and name not in names:
                names.append(name)
            elif sign == "-" and name in names:
                names.remove(name)
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
]
