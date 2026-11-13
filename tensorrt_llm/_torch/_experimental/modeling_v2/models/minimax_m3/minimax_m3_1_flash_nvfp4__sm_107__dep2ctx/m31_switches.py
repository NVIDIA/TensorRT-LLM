# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kill switches of the MiniMax-M3.1 dep2ctx target's default-on items (each target keeps its own copy: targets
share only the catalog). The target itself is on by default for its role (routing.py).

Every item is on by default; ``TRTLLM_M31_DISABLE_<ID>=1`` turns item ``<ID>`` off and restores the path it replaced
(``SWITCHES`` lists each id and what it restores). The environment is read once, at import, so every rank and every
step of a process sees the same setting (a step must not change its graph mid-run).

Values (case-insensitive, surrounding blanks ignored): ``1`` / ``true`` / ``on`` / ``yes`` set a switch; empty / ``0`` /
``false`` / ``off`` / ``no`` leave it unset; anything else raises at import, naming the variable, so a typo fails loudly
instead of silently turning an item on or off. ``m31_disabled`` raises on an id that is not in ``SWITCHES``; a set
variable whose id is not in ``SWITCHES`` is reported once at import and ignored.
"""

import os
from typing import Dict, FrozenSet

from tensorrt_llm.logger import logger

_TARGET = "dep2ctx"
_PREFIX = "TRTLLM_M31_DISABLE_"
_SET = frozenset(("1", "true", "on", "yes"))
_UNSET = frozenset(("", "0", "false", "off", "no"))

#: The target's items: id -> what ``TRTLLM_M31_DISABLE_<id>=1`` restores.
SWITCHES: Dict[str, str] = {
    "CTX_MOE_TAIL": "the MegaMoE launch's own top-k reduce and the separate shared-expert add",
    "DEP2CTX_A1_FUSED_IN_BCG": "the stock QKV projection under a breakable prefill graph capture",
    "DEP2CTX_A1_IN_BCG": "the stock QKV projection and producer under a breakable prefill graph capture",
    "DEP2CTX_BCG_LARGE": "the prefill graph capture capped at the base bucket",
    "DEP2CTX_BCG_PAD_MASK": "graph padding rows routed to experts like live rows",
    "DEP2CTX_BCG_RAW_BREAK": "the attention break through the torch custom-op dispatcher",
    "DEP2CTX_C1_BCG": "o_proj quantizing its input itself under a breakable prefill graph capture",
    "DEP2CTX_COMBINE_DROP_BF16": "the combine's BF16 store of the attention output",
    "DEP2CTX_COMBINE_MXFP8": "o_proj quantizing the sparse prefill combine's output itself",
    "DEP2CTX_CUTEDSL_MOE": "the routed experts off the CuTe DSL backend",
    "DEP2CTX_GATE_CAST": "aten's dtype-converting copy for the router input",
    "DEP2CTX_IDX_SCORER": "the fmha_sm100 OnlyScore proxy pass for the prefill indexer scores",
    "DEP2CTX_ITEMS_IN_BCG": "the stock paths of the boundary items under a breakable prefill graph capture",
    "DEP2CTX_K2Q_LPT_REUSE": "the LPT k2q CSR build allocating its buffers per layer",
    "DEP2CTX_LARGE_PREFILL_GRAPHS": "every item that rides on the prefill breakable-graph buckets above 4096 tokens",
    "DEP2CTX_MOE_RUNNER_CACHE": "the routed experts' FC1 / FC2 through torch.library dispatch per call",
    "DEP2CTX_MXFP8_CUBLASLT": "the CUTLASS MXFP8 GEMM for the prefill MXFP8 GEMMs",
    "DEP2CTX_NORMQ": "the stock residual add + RMSNorm at the prefill layer boundary",
    "DEP2CTX_NORMQ_F32": "the plain boundary op and the gate's own FP32 cast",
    "DEP2CTX_NORMQ_FP4": "the routed experts quantizing their NVFP4 input themselves",
    "DEP2CTX_NORMQ_STAGE": "the routed experts' own route-quant launch",
    "DEP2CTX_NORMQ_STAGE_DROP_NORM": "the BF16 norm always written at a staged boundary",
    "DEP2CTX_NORMQ_V2": "the served boundary op instead of the ring kernel",
    "DEP2CTX_PROXY_Q128": "the engine's prefill proxy plan (no 128-row query tiles)",
    "DEP2CTX_QKV_FUSED": "the stock MXFP8 QKV GEMM and the separate producer",
    "DEP2CTX_QKV_ROWS": "the per-token NVFP4 QKV + indexer producer",
    "DEP2CTX_ROUTER_BF16S": "the TF32 router on the boundary's FP32 copy",
    "DEP2CTX_SCHED_LPT": "the k2q CSR schedule in its stock order",
    "DEP2CTX_SPARSE_OUT": "the sparse prefill combine's copied output",
    "DEP2CTX_SWIGLU_MXFP8": "down_proj quantizing the SwiGLU output itself",
    "DEP2CTX_SWIGLU_NOH": "the SwiGLU's bf16 activation always written",
    "DEP2CTX_TOPK_CAPTURE": "a speculative capture layer declining the top-k fold",
    "DEP2CTX_TOPK_FOLD": "the separate top-k reduce + shared add and the stock boundary",
    "DEP2CTX_TSEL": "the built-in prefill top-16 block select",
    "DEP2CTX_TSEL_ROWS": "the row-major tile select off (the exact prefill scorer then stays off too)",
    "DEP2CTX_UNPERMUTE_ADD": "the separate shared-expert add after the unpermute",
    # Read by the family DSpark prefill store (ops/dspark/dspark_prefill_store) through the callable the target passes.
    "DEP2CTX_DSPARK_STORE": "the stock drafter prefill store (every store item off)",
    "DEP2CTX_DSPARK_STORE_FOLD": "the drafter's k_norm weight unfolded",
    "DRAFTER_LEAN_GLUE": "the drafter's stock strided-view glue",
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


# The item ids switched off in this process, read once at import.
_DISABLED: FrozenSet[str] = frozenset(
    name[len(_PREFIX) :]
    for name, value in os.environ.items()
    if name.startswith(_PREFIX) and _is_set(name, value)
)

# Set switches this registry does not list: one environment may serve several targets, the family modules and the
# engine, which read their own, so they are reported once, not refused.
_FOREIGN = sorted(_DISABLED - SWITCHES.keys())
if _FOREIGN:
    logger.warning(
        f"[M3.1 {_TARGET}] set switches this target does not read (the family, the engine or another target may): "
        + ", ".join(_PREFIX + item for item in _FOREIGN)
    )


def m31_disabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_DISABLE_<item_id>`` turned the item off for this process. Raises ``KeyError`` for an id
    that is not in ``SWITCHES``."""
    if item_id not in SWITCHES:
        raise KeyError(
            f"{item_id!r} is not a switch of the {_TARGET} target (see m31_switches.SWITCHES)"
        )
    return item_id in _DISABLED


def large_prefill_graphs_disabled(item_id: str) -> bool:
    """Whether a large-prefill-graph item (prefill breakable-graph buckets above 4096 and the items that ride on them)
    is off: ``TRTLLM_M31_DISABLE_DEP2CTX_LARGE_PREFILL_GRAPHS=1`` turns every such item off,
    ``TRTLLM_M31_DISABLE_<item_id>=1`` that one."""
    return m31_disabled("DEP2CTX_LARGE_PREFILL_GRAPHS") or m31_disabled(item_id)


# Default-off items (opt-ins): ``TRTLLM_M31_ENABLE_<ID>=1`` turns item ``<ID>`` on. Each changes the target's numerics
# against the default path (not bitwise) or is not faster on every shape, so it stays off unless asked for. Same strict
# values as the kill switches, read once at import; an opt-in variable naming an id this target does not register is
# logged once and otherwise ignored.
_ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"

# Opt-in id -> what TRTLLM_M31_ENABLE_<id>=1 turns on.
OPT_INS: Dict[str, str] = {
    "DEP2CTX_IDX_SCORER_K64": "the prefill scorer built at MMA K = 64 (TRTLLM_M31_IDX_SCORER_MMA_K=64; not bitwise)",
}


def _read_enabled() -> FrozenSet[str]:
    """The registered opt-in ids switched on in this process; logs each unregistered opt-in variable once."""
    out = set()
    for name, value in os.environ.items():
        if not name.startswith(_ENABLE_PREFIX):
            continue
        item_id = name[len(_ENABLE_PREFIX) :]
        if item_id not in OPT_INS:
            logger.warning(f"{name} is not an opt-in of this MiniMax-M3.1 target; ignored")
            continue
        if _is_set(name, value):
            out.add(item_id)
    return frozenset(out)


# The opt-in ids switched on in this process, read once at import.
_ENABLED: FrozenSet[str] = _read_enabled()


def m31_enabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_ENABLE_<item_id>`` turned the default-off item on for this process."""
    if item_id not in OPT_INS:
        raise KeyError(
            f"{item_id!r} is not a registered opt-in of this target (m31_switches.OPT_INS)"
        )
    return item_id in _ENABLED
