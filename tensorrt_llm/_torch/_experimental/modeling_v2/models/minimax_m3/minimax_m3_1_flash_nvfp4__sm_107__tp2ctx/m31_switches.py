# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kill switches of the MiniMax-M3.1 tp2ctx target's default-on items (each target keeps its own copy: targets
share only the catalog).

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

_TARGET = "tp2ctx"
_PREFIX = "TRTLLM_M31_DISABLE_"
_SET = frozenset(("1", "true", "on", "yes"))
_UNSET = frozenset(("", "0", "false", "off", "no"))

#: The target's items: id -> what ``TRTLLM_M31_DISABLE_<id>=1`` restores.
SWITCHES: Dict[str, str] = {
    "TP2CTX_A1_IN_BCG": "the stock QKV projection and producer under a breakable prefill graph capture",
    "TP2CTX_COMBINE_DROP_BF16": "the combine's BF16 store of the attention output",
    "TP2CTX_COMBINE_MXFP8": "o_proj quantizing the sparse prefill combine's output itself",
    "TP2CTX_CUTEDSL_MOE": "the routed experts off the CuTe DSL backend",
    "TP2CTX_GATE_CAST": "aten's dtype-converting copy for the router input",
    "TP2CTX_GEN_FINALIZE": "the routed experts' own finalize and the separate shared-expert add",
    "TP2CTX_ITEMS_IN_BCG": "the stock paths of the boundary items under a breakable prefill graph capture",
    "TP2CTX_MOE_RUNNER_CACHE": "the routed experts' FC1 / FC2 through torch.library dispatch per call",
    "TP2CTX_MXFP8_CUBLASLT": "the CUTLASS MXFP8 GEMM for the prefill MXFP8 GEMMs",
    "TP2CTX_NORMQ": "the fused all-reduce + RMSNorm at the prefill layer boundary",
    "TP2CTX_NORMQ_F32": "the plain boundary op and the gate's own FP32 cast",
    "TP2CTX_NORMQ_FP4": "the routed experts quantizing their NVFP4 input themselves",
    "TP2CTX_PROXY_Q128": "the engine's prefill proxy plan (no 128-row query tiles)",
    "TP2CTX_QKV_FUSED": "the stock MXFP8 QKV GEMM and the separate producer",
    "TP2CTX_QKV_ROWS": "the per-token NVFP4 QKV + indexer producer",
    "TP2CTX_SCHED_LPT": "the k2q CSR schedule in its stock order",
    "TP2CTX_SP": "the full-T all-reduce boundaries (no sequence-parallel boundaries)",
    "TP2CTX_SP_ADDNORM": "the owned-row peer add and the boundary op as two kernels",
    "TP2CTX_SP_ADDOUT": "the MoE output copied into the sequence-parallel boundary's input slot",
    "TP2CTX_SP_CAPTURE": "the full-T boundary on a speculative capture layer",
    "TP2CTX_SP_LOGITS": "the full-T pre-MoE boundary for experts that route in their kernel",
    "TP2CTX_SP_OVL": "the pre-attention push waited for before the fused QKV",
    "TP2CTX_SP_PMOVL": "the pre-MoE products pushed after the routing on the current stream",
    "TP2CTX_SPARSE_OUT": "the sparse prefill combine's copied output",
    "TP2CTX_SWIGLU_MXFP8": "down_proj quantizing the SwiGLU output itself",
    "TP2CTX_SWIGLU_NOH": "the SwiGLU's bf16 activation always written",
    "TP2CTX_TSEL": "the built-in prefill top-16 block select",
    "TP2CTX_UNPERMUTE_ADD": "the separate shared-expert add after the unpermute",
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
