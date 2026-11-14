# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kill switches of the default-on items in the MiniMax-M3.1-Flash family modules (``ops/``, ``kernels/``, ``msa/``).

Family modules may not import a target, so the items they own read this registry instead of a target's
``m31_switches.py``. The parser and the values are the targets': every item is on by default and
``TRTLLM_M31_DISABLE_<ID>=1`` turns item ``<ID>`` off and restores the path it replaced. The environment is read once,
at import, so every rank and every step of a process sees the same setting.

Values (case-insensitive, surrounding blanks ignored): ``1`` / ``true`` / ``on`` / ``yes`` set a switch; empty / ``0`` /
``false`` / ``off`` / ``no`` leave it unset; anything else raises at import, naming the variable. ``m31_disabled``
raises on an id that is not listed in ``SWITCHES``. Variables with the same prefix that name a target's items are
left to that target's registry.
"""

import os
from typing import Dict, FrozenSet

_PREFIX = "TRTLLM_M31_DISABLE_"
_SET = frozenset(("1", "true", "on", "yes"))
_UNSET = frozenset(("", "0", "false", "off", "no"))

#: The family's items: id -> what ``TRTLLM_M31_DISABLE_<id>=1`` restores.
SWITCHES: Dict[str, str] = {
    "DEP_IDXK_NATIVE_LAYOUT": "msa/kernels/nvfp4_index_k: the reader's row-contiguous index-K page layout",
    "DRAFT_PROB1": "kernels/m3ll/prob1/draft_prob1: the in-tree DSpark draft (no fused probabilistic draft op)",
    "DSPARK_CTX_STORE_CPP": "ops/dspark/dspark_ctx_store: the stock drafter context store",
    "DSPARK_GLUE_CPP": "ops/dspark/dspark_ctx_advance and dspark_forward_indices: the stock drafter glue lines",
    "DSPARK_MARKOV_CPP": "ops/dspark/dspark_markov_bias: the in-tree Markov bias lines",
    "MXFP8_CUTE_PIN": "ops/gemm/mxfp8_cute_pinned_tactics: FlashInfer's default MXFP8 CuTe DSL GEMM tactic",
    "NATIVE_ADD_RMSNORM": "ops/add_rmsnorm: the CuTe DSL add + RMSNorm kernels (FlashInfer for the plain form)",
    "NATIVE_MARKOV_SAMPLING": "ops/markov_sampling: the torch Markov sampling lines",
    "NATIVE_SWIGLU": "ops/native_swiglu: the stock SwiGLU",
    "NVFP4_COMPACT_NOSYNC": "msa/kernels/nvfp4_index_k: the pageable offset copy of the compaction",
    "ROUTER_HALF_PIN": "ops/gemm/router_half_pin: the full-row router GEMM",
    "TP2CTX_ROUTER_ZC": "ops/comm/tp2ctx_ipc: the per-call zero fill + copy of the router's full-step input",
    "TP2CTX_SP": "ops/comm/tp2ctx_ipc: the all-reduce prefill boundary",
}


def _is_set(name: str, value: str) -> bool:
    """Whether the switch variable ``name`` is set to ``value``; raises on a value of neither kind."""
    word = value.strip().lower()
    if word in _SET:
        return True
    if word in _UNSET:
        return False
    raise ValueError(
        f"{name}={value!r}: expected 1 / true / on / yes (set) or 0 / false / off / no / empty (unset)"
    )


# The family item ids switched off in this process, read once at import.
_DISABLED: FrozenSet[str] = frozenset(
    item for item in SWITCHES if _is_set(_PREFIX + item, os.environ.get(_PREFIX + item, ""))
)


def m31_disabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_DISABLE_<item_id>`` turned the family item off for this process."""
    if item_id not in SWITCHES:
        raise KeyError(
            f"{item_id!r} is not a MiniMax-M3.1 family switch (see m31_switches.SWITCHES)"
        )
    return item_id in _DISABLED
