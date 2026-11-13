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
from .flashinfer_trtllm_gen import FlashInferTrtllmGenFmha
from .fp4_mla import Fp4MlaFmha
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
    from .flash_mla import FlashMlaFmha
    from .flashinfer_sparse_mla import FlashInferSparseMlaFmha
    from .fmha_v2 import FmhaV2Fmha
    from .gqa_decode import (
        GqaDecodeFmha,
        GqaKvReplicaFmha,
        GqaKvReplicaFusedFmha,
        GqaSegmentedFmha,
        GqaSplitKvFmha,
    )
    from .mmha import MmhaFmha
    from .msa_decode import MsaDecodeFmha
    from .msa_prefill import MsaPrefillFmha
    from .prims_ts_block_sparse import PrimsTSBlockSparseFmha
    from .prims_ts_sparse_mla import PrimsTSSparseMlaFmha
    from .trtllm_gen import TrtllmGenFmha
    from .unfused import UnfusedFmha
    from .xqa import XqaFmha

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
        # GQA decode layers (GqaDecodeSetup) only: dedicated generation kernels in order of precedence, then
        # trtllm-gen with the module's KV append / deferred prologue.
        "gqa_kv_replica_fused": GqaKvReplicaFusedFmha,
        "gqa_kv_replica": GqaKvReplicaFmha,
        "gqa_split_kv": GqaSplitKvFmha,
        "gqa_segmented": GqaSegmentedFmha,
        "gqa_decode": GqaDecodeFmha,
        "flashinfer_trtllm_gen": FlashInferTrtllmGenFmha,
        # The native attention op's libraries. One order serves MHA and MLA:
        # fmha_v2 serves no MHA generation and XQA no context phase.
        "trtllm_gen": TrtllmGenFmha,
        "flash_mla": FlashMlaFmha,
        "xqa": XqaFmha,
        "fmha_v2": FmhaV2Fmha,
        "mmha": MmhaFmha,
        "unfused": UnfusedFmha,
    }


FMHA_LIBS: dict[str, FmhaCls] = init_fmha_libs()
_OPT_IN_FMHA_LIBS: frozenset[str] = frozenset({"prims_ts"})
DEFAULT_FMHA_LIBS: tuple[str, ...] = tuple(
    name for name in FMHA_LIBS if name not in _OPT_IN_FMHA_LIBS
)
# Names in TLLM_FMHA_LIBS that stand for several libraries.
FMHA_LIB_GROUPS: dict[str, tuple[str, ...]] = {
    "fallback": ("trtllm_gen", "flash_mla", "xqa", "fmha_v2", "mmha", "unfused"),
}


def _expand_fmha_lib_name(name: str) -> tuple[str, ...]:
    """The libraries one TLLM_FMHA_LIBS entry names: itself, or a group's members."""
    if name in FMHA_LIB_GROUPS:
        return FMHA_LIB_GROUPS[name]
    if name not in FMHA_LIBS:
        raise ValueError(f"Unknown FMHA library {name!r} in TLLM_FMHA_LIBS.")
    return (name,)


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
            members = _expand_fmha_lib_name(name)
            if sign == "+":
                enabled_names.update(members)
            else:
                enabled_names.difference_update(members)
        names = [name for name in FMHA_LIBS if name in enabled_names]
    else:
        names = []
        for token in tokens:
            for name in _expand_fmha_lib_name(token):
                if name not in names:
                    names.append(name)

    return tuple(names)


def get_enabled_fmha_lib_classes() -> list[FmhaCls]:
    return [FMHA_LIBS[name] for name in _parse_fmha_libs_env()]


__all__ = [
    "DEFAULT_FMHA_LIBS",
    "FMHA_LIB_GROUPS",
    "FMHA_LIBS",
    "FmhaCls",
    "get_enabled_fmha_lib_classes",
    "init_fmha_libs",
]
