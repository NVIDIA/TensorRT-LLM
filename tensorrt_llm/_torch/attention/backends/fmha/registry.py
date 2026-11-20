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

import functools
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
    "prims_ts",
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


def _parse_fmha_libs_env() -> tuple[tuple[str, str], ...]:
    return _parse_fmha_libs_value(os.environ.get("TLLM_FMHA_LIBS", ""))


@functools.lru_cache(maxsize=8)
def _parse_fmha_libs_value(value: str) -> tuple[tuple[str, str], ...]:
    """Cache the syntax of environment overrides independently of registration."""
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
        entries.append((sign, name))
    return tuple(entries)


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
    entries = _parse_fmha_libs_env() if use_fmha_libs_env else ()
    return _resolve_fmha_lib_classes(fmha_libs, entries, prepended_fmha_lib_classes)


def _resolve_fmha_lib_classes(
    fmha_libs: Sequence[str | FmhaCls] | None,
    env_entries: tuple[tuple[str, str], ...],
    prepended_fmha_lib_classes: Sequence[FmhaCls] = (),
) -> list[FmhaCls]:
    entries = []
    for sign, name in env_entries:
        if name not in FMHA_LIBS:
            raise ValueError(f"Unknown FMHA library {name!r} in TLLM_FMHA_LIBS.")
        entries.append((sign, FMHA_LIBS[name]))
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


@functools.lru_cache(maxsize=8)
def _global_fmha_lib_classes(entries: tuple[tuple[str, str], ...]) -> tuple[FmhaCls, ...]:
    return tuple(_resolve_fmha_lib_classes(None, entries))


def is_fmha_lib_enabled(name: str) -> bool:
    """Whether ``TLLM_FMHA_LIBS`` enables the library registered as ``name``."""
    return FMHA_LIBS.get(name) in _global_fmha_lib_classes(_parse_fmha_libs_env())


@functools.lru_cache(maxsize=8)
def _step_planning_classes(entries: tuple[tuple[str, str], ...]) -> tuple[FmhaCls, ...]:
    return tuple(cls for cls in _global_fmha_lib_classes(entries) if cls.plans_steps())


@functools.lru_cache(maxsize=8)
def _step_stats_classes(entries: tuple[tuple[str, str], ...]) -> tuple[FmhaCls, ...]:
    return tuple(cls for cls in _global_fmha_lib_classes(entries) if cls.step_stats_field is not None)


def step_planning_fmha_lib_classes() -> tuple[FmhaCls, ...]:
    """Enabled libraries that plan every step (``Fmha.plan_step``); cached, prepare() calls it."""
    return _step_planning_classes(_parse_fmha_libs_env())


def step_stats_fmha_lib_classes() -> tuple[FmhaCls, ...]:
    """Enabled libraries that keep per-step counters (``Fmha.step_stats``)."""
    return _step_stats_classes(_parse_fmha_libs_env())


def fmha_step_stats(kv_cache_manager: object) -> dict[str, dict[str, int]] | None:
    """Cumulative per-step counters of the enabled libraries that keep them, for the steps attended
    on ``kv_cache_manager``'s cache, by iteration-stats record field; None when no such library is
    enabled. Cumulative, so a consumer that misses records still reads exact totals."""
    classes = step_stats_fmha_lib_classes()
    if not classes or kv_cache_manager is None:
        return None
    return {cls.step_stats_field: cls.step_stats(kv_cache_manager) for cls in classes}


def reset_fmha_step_stats(kv_cache_manager: object) -> None:
    """Zero the counters of ``kv_cache_manager`` (the executor drops its warmup steps)."""
    if kv_cache_manager is not None:
        for cls in step_stats_fmha_lib_classes():
            cls.reset_step_stats(kv_cache_manager)


__all__ = [
    "DEFAULT_FMHA_LIBS",
    "FMHA_LIBS",
    "FmhaCls",
    "fmha_lib",
    "fmha_step_stats",
    "get_enabled_fmha_lib_classes",
    "is_fmha_lib_enabled",
    "reset_fmha_step_stats",
    "step_planning_fmha_lib_classes",
    "step_stats_fmha_lib_classes",
]
