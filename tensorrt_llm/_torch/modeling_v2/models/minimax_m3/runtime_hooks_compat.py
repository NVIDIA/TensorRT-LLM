# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``RuntimeHooks`` of the MiniMax-M3.1-Flash targets, built for the engine this process runs.

The M3.1 targets request step-loop hooks by field name, and some of those fields are not (yet) fields of this engine's
``RuntimeHooks`` or ``MsaRuntimePolicy``. Both dataclasses are frozen, so passing such a field would raise. ``build``
passes every requested field the engine has, and handles each one it lacks by what running without it costs:

* ``MAIN_DEFAULT``: the engine already behaves this way by default (E4M3 saturating KV-cache writes, a zeroed and
  exactly sized MSA KV pool). Running without the field is the same, unless the process turned that default off
  (``missing_defaults``), which declines the target.
* ``PERF_ONLY`` / ``MSA_PERF_ONLY``: the same tokens, with more host or device work per step.
* ``DRAFT_ONLY``: the drafts, so the acceptance length, may change; the verified tokens do not.
* ``CONDITIONAL``: needed only by a configuration the family's ``route`` declines while the engine lacks it.
* ``REQUIRED``: the target is not correct without it, so requesting it on an engine without it raises ``NoTarget``.

A requested field the engine lacks and none of these sets names is a programming error (``TypeError``), so every new
request is classified before it ships. Each dropped field that was requested on is logged once per target. Field
names pass through unchanged: once the engine gains a field, the same request reaches it.

This module imports nothing from any target, so ``route`` may import it.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Set, Tuple

from tensorrt_llm._torch.pyexecutor.runtime_hooks import MsaRuntimePolicy, RuntimeHooks
from tensorrt_llm.logger import logger

from ..._router_index import NoTarget

__all__ = [
    "CONDITIONAL",
    "DRAFT_ONLY",
    "MAIN_DEFAULT",
    "MSA_PERF_ONLY",
    "MsaRuntimePolicy",
    "PERF_ONLY",
    "REQUIRED",
    "RuntimeHooks",
    "build",
    "has_hook",
    "missing_defaults",
    "requested",
]

MAIN_DEFAULT: FrozenSet[str] = frozenset({"fp8_saturating_casts", "msa_kv_cache_fixes"})
PERF_ONLY: FrozenSet[str] = frozenset(
    {
        "context_onehot_index_fill",
        "context_token_array",
        "defer_gen_reuse_commit",
        "dflash_host_ctx_alloc",
        "install_tactic_pins",
        "kv_gpu_prefetch",
        "msa_prefill_fork",
        "pack_cpp_scatter",
        "patch_unchanged_draft_block_offsets",
        "persist_flashinfer_mxfp8_tactics",
        "prefill_capture_extra_min_top",
        "prefill_capture_extra_num_tokens",
        "prefill_capture_max_num_tokens",
        "proxy_ms_reuse",
        "sparse_attention",
        "skip_accepted_draft_upload",
        "skip_unchanged_draft_block_offsets",
        "spec_batch_map_changed_only",
        "spec_slot_ids_changed_only",
        "spec_tail_bookkeep",
        "token_array_all_requests",
    }
)
DRAFT_ONLY: FrozenSet[str] = frozenset({"disagg_draft_kv", "draft_kv_reserve_guard"})
# The separate draft pool's history-first resize; ``route`` declines the configuration that needs it.
CONDITIONAL: FrozenSet[str] = frozenset({"draft_pool_fixes"})
# The release of a model's prefill CUDA graphs, which resets the pointer guard of the graphs that baked the addresses.
REQUIRED: FrozenSet[str] = frozenset({"prefill_graphs_released"})
MSA_PERF_ONLY: FrozenSet[str] = frozenset(
    {
        "block_ids_from_views",
        "forward_staging",
        "kv_update_tail",
        "long_family_min_total_tokens",
        "prefill_proxy_kv_splits",
        "prefill_proxy_kv_splits_by_rows",
        "prefill_proxy_qo_tile",
    }
)
_CLASSIFIED = MAIN_DEFAULT | PERF_ONLY | DRAFT_ONLY | CONDITIONAL | REQUIRED

# The engine's zero-initialized MSA KV pool, which stands in for ``msa_kv_cache_fixes``; its switch reads "1" as on.
_KV_POOL_ZERO_INIT_ENV = "TRTLLM_M3_KV_POOL_ZERO_INIT"

_LOGGED: Set[str] = set()
# What each target last asked ``build`` for, before any field was dropped (``requested``).
_REQUESTED: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}


def _fields(cls: type) -> FrozenSet[str]:
    return frozenset(field.name for field in dataclasses.fields(cls))


def has_hook(name: str) -> bool:
    """Whether this engine's ``RuntimeHooks`` has the field ``name``."""
    return name in _fields(RuntimeHooks)


def missing_defaults() -> List[str]:
    """The engine defaults that stand in for a ``MAIN_DEFAULT`` field this engine lacks but this process turned off.

    Empty when the targets run correctly on this engine as far as those fields go.
    """
    missing = []
    if not has_hook("msa_kv_cache_fixes") and os.environ.get(_KV_POOL_ZERO_INIT_ENV, "1") != "1":
        missing.append(
            f"{_KV_POOL_ZERO_INIT_ENV}={os.environ[_KV_POOL_ZERO_INIT_ENV]} (the MSA KV pool must be zero-initialized)"
        )
    return missing


def requested(target: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The hook and MSA policy fields ``target`` last passed to ``build``, including those this engine lacks."""
    hooks, msa = _REQUESTED[target]
    return dict(hooks), dict(msa)


def _enabled(value: Any) -> bool:
    """Whether a requested value turns its hook on (a hook's default is None, False, 0 or empty)."""
    if value is None or value is False:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value != 0
    if isinstance(value, (tuple, list, dict, set, frozenset)):
        return len(value) > 0
    return True


def build(
    hooks: Mapping[str, Any], msa: Optional[Mapping[str, Any]] = None, *, target: str
) -> RuntimeHooks:
    """``RuntimeHooks(**hooks, msa=MsaRuntimePolicy(**msa))`` with the fields this engine lacks handled as the module
    docstring describes.

    Raises ``TypeError`` for a field the engine lacks that is not classified, and ``NoTarget`` when a ``REQUIRED``
    field is requested on, or when a ``MAIN_DEFAULT`` field's stand-in default is turned off (``missing_defaults``).
    """
    if "msa" in hooks:
        raise TypeError(f"{target}: pass the MSA policy fields as msa=, not inside hooks")
    msa = dict(msa or {})
    _REQUESTED[target] = (dict(hooks), dict(msa))
    have, msa_have = _fields(RuntimeHooks), _fields(MsaRuntimePolicy)
    unknown = sorted(set(hooks) - have)
    msa_unknown = sorted(set(msa) - msa_have)
    unclassified = [name for name in unknown if name not in _CLASSIFIED]
    unclassified += [f"msa.{name}" for name in msa_unknown if name not in MSA_PERF_ONLY]
    if unclassified:
        raise TypeError(
            f"{target}: runtime hook fields this engine does not have and runtime_hooks_compat does not classify: "
            f"{', '.join(unclassified)}"
        )
    required = [name for name in unknown if name in REQUIRED and _enabled(hooks[name])]
    if required:
        raise NoTarget(f"{target} needs runtime hooks this engine lacks: {', '.join(required)}")
    if any(name in MAIN_DEFAULT and _enabled(hooks[name]) for name in unknown):
        defaults_off = missing_defaults()
        if defaults_off:
            raise NoTarget(
                f"{target} needs the engine defaults this process turned off: {'; '.join(defaults_off)}"
            )

    dropped: Dict[str, List[str]] = {}
    for name in unknown:
        if _enabled(hooks[name]):
            for kind, names in (
                ("the engine default", MAIN_DEFAULT),
                ("perf only", PERF_ONLY),
                ("drafts only", DRAFT_ONLY),
                ("declined by route where needed", CONDITIONAL),
            ):
                if name in names:
                    dropped.setdefault(kind, []).append(name)
    for name in msa_unknown:
        if _enabled(msa[name]):
            dropped.setdefault("perf only", []).append(f"msa.{name}")
    if dropped and target not in _LOGGED:
        _LOGGED.add(target)
        logger.info(
            f"[M3.1 hooks] {target}: running without the hooks this engine does not have: "
            + "; ".join(f"{kind}: {', '.join(names)}" for kind, names in dropped.items())
        )
    return RuntimeHooks(
        **{name: value for name, value in hooks.items() if name in have},
        msa=MsaRuntimePolicy(**{name: value for name, value in msa.items() if name in msa_have}),
    )
