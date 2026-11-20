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

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TypeAlias, TypeVar

from .interface import Fmha

FmhaCls: TypeAlias = type[Fmha]
_FmhaT = TypeVar("_FmhaT", bound=Fmha)

FMHA_LIBS: dict[str, FmhaCls] = {}
DEFAULT_FMHA_LIBS = (
    "fp4_mla",
    "triton_custom_mask",
    "cute_dsl_mla",
    "msa_decode",
    "msa_prefill",
    "flashinfer_sparse_mla",
    "prims_ts_block_sparse",
    "prims_ts_sparse_mla",
    "prims_ts_dsa_sparse_mla",
    # GQA generation kernels in precedence order, then general GQA decode.
    "gqa_kv_replica_fused",
    "gqa_kv_replica",
    "gqa_split_kv",
    "gqa_segmented",
    "gqa_decode",
    "flashinfer_trtllm_gen",
    "fallback",
)


def fmha_lib(name: str) -> Callable[[type[_FmhaT]], type[_FmhaT]]:
    """Register an FMHA class for name lookup without changing the default order."""
    if not name or name != name.strip() or name[0] in "+-" or "," in name:
        raise ValueError(f"Invalid FMHA library name {name!r}.")

    def register(cls: type[_FmhaT]) -> type[_FmhaT]:
        if FMHA_LIBS.setdefault(name, cls) is not cls:
            raise ValueError(f"FMHA library {name!r} is already registered.")
        return cls

    return register


def _get_fmha_lib_class(name: str) -> FmhaCls:
    try:
        return FMHA_LIBS[name]
    except KeyError:
        raise ValueError(f"Unknown FMHA library {name!r}.") from None


def _parse_fmha_libs_env() -> list[tuple[str, FmhaCls]]:
    value = os.environ.get("TLLM_FMHA_LIBS", "")
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    has_delta = any(token[0] in "+-" for token in tokens)
    if has_delta and not all(token[0] in "+-" for token in tokens):
        raise ValueError(
            "TLLM_FMHA_LIBS must use either an exact comma-separated list "
            "or only +/- delta entries."
        )

    entries = []
    for token in tokens:
        sign = token[0] if has_delta else ""
        name = token[1:].strip() if has_delta else token
        if not name:
            raise ValueError(f"Invalid empty FMHA library entry in {value!r}.")
        if name not in FMHA_LIBS:
            raise ValueError(f"Unknown FMHA library {name!r} in TLLM_FMHA_LIBS.")
        entries.append((sign, FMHA_LIBS[name]))
    return entries


def get_enabled_fmha_lib_classes(
    fmha_libs: Sequence[str | FmhaCls] | None = None,
    use_fmha_libs_env: bool = True,
    prepended_fmha_lib_classes: Sequence[FmhaCls] = (),
) -> list[FmhaCls]:
    """Resolve ordered candidates and apply environment overrides.

    Args:
        fmha_libs: Registered names and classes. None selects the default list.
        use_fmha_libs_env: Apply deltas or an exact list from TLLM_FMHA_LIBS.
        prepended_fmha_lib_classes: Model classes to place before the base candidates.
    """
    entries = _parse_fmha_libs_env() if use_fmha_libs_env else []
    if entries and not entries[0][0]:
        return list(dict.fromkeys(cls for _, cls in entries))

    candidates = DEFAULT_FMHA_LIBS if fmha_libs is None else fmha_libs
    classes = [_get_fmha_lib_class(lib) if isinstance(lib, str) else lib for lib in candidates]
    additions: dict[FmhaCls, None] = {}
    excluded: set[FmhaCls] = set()
    for sign, cls in entries:
        if sign == "+":
            excluded.discard(cls)
            additions[cls] = None
        else:
            additions.pop(cls, None)
            excluded.add(cls)

    ordered = dict.fromkeys((*additions, *prepended_fmha_lib_classes, *classes))
    return [cls for cls in ordered if cls not in excluded]


def is_fmha_lib_enabled(name: str) -> bool:
    """Whether ``TLLM_FMHA_LIBS`` enables the library registered as ``name``."""
    return FMHA_LIBS.get(name) in get_enabled_fmha_lib_classes()


__all__ = [
    "DEFAULT_FMHA_LIBS",
    "FMHA_LIBS",
    "FmhaCls",
    "fmha_lib",
    "get_enabled_fmha_lib_classes",
    "is_fmha_lib_enabled",
]
