# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kill switches of this MiniMax-M3.1 target's default-on items (each target keeps its own copy: targets share
only the catalog).

Every item is on by default; ``TRTLLM_M31_DISABLE_<ID>=1`` turns item ``<ID>`` off and restores the path it replaced.
The environment is read once, at import, so every rank and every step of a process sees the same setting (a step must
not change its graph mid-run). ``ITEMS`` registers every id this target reads, with what its switch restores;
``m31_disabled`` raises on an id that is not registered.

Values (case-insensitive, surrounding blanks ignored): ``1`` / ``true`` / ``on`` / ``yes`` set a switch; empty / ``0`` /
``false`` / ``off`` / ``no`` leave it unset; anything else raises at import, naming the variable, so a typo fails loudly
instead of silently turning an item on or off. A ``TRTLLM_M31_DISABLE_*`` variable naming an id this target does not
register (another target's switch in a shared environment, or a renamed id) is logged once and otherwise ignored.
"""

import os
from typing import Dict, FrozenSet

from tensorrt_llm.logger import logger

_PREFIX = "TRTLLM_M31_DISABLE_"
_SET = frozenset(("1", "true", "on", "yes"))
_UNSET = frozenset(("", "0", "false", "off", "no"))

# Item id -> what TRTLLM_M31_DISABLE_<id>=1 restores.
ITEMS: Dict[str, str] = {
    "AR64AUTO": "the configured all-reduce strategy at the 49..64-row decode boundaries",
    "BATCH_CHANGED_ONLY": "the every-step copies of the drafter's batch indices and batch-to-slot map",
    "BLOCKOFF_GUARD": "the drafter KV cache manager's block-table copy on every step",
    "BOUNDARY_PACK": "the unpacked batch-1 layer-boundary step inputs",
    "CTXNORMQ": "the fused all-reduce + norm and the consumers' own quantize at large prefill layer boundaries",
    "DRAFTER_BLOCK_TABLE_REFRESH_FUSED": "the drafter's unfused block-table refresh",
    "DRAFTER_BLOCK_TABLE_REFRESH_GRID": "the serial block-table refresh kernel (not the row x column-chunk grid)",
    "DRAFTER_CTX_ADVANCE_FUSED": "the drafter's unfused context advance",
    "DRAFTER_CTX_KV_STORE_FUSED": "the drafter's unfused context KV store",
    "DRAFTER_FORWARD_INDICES_FUSED": "the drafter's unfused forward indices",
    "DRAFTER_FUSED_GLUE": "every fused drafter glue launch (DRAFTER_*_FUSED, DRAFTER_BLOCK_TABLE_REFRESH_GRID)",
    "DRAFTER_INPUTS_FUSED": "the drafter's unfused input preparation",
    "DRAFTER_LEAN_GLUE": "the drafter glue without the strided views",
    "DRAFTER_MXFP8": "the drafter's BF16 GEMMs (instead of the MXFP8 GEMV)",
    "DRAFT_APPEND_FUSE": (
        "the drafter's per-layer paged K / V append (instead of writing it in the q / k norm + RoPE "
        "launch)"
    ),
    "DRAFT_ATTN_PDL": "the drafter attention without programmatic dependent launch",
    "DRAFT_KV_REWIND_FUSED": "the in-tree draft-forward KV rewind bookkeeping",
    "DRAFT_S4": "the in-tree Markov chain (instead of the one-launch sharded chain)",
    "EX11": "flashinfer's three launches for the drafter q / k RMSNorm + partial RoPE",
    "FUSED_CAPTURE": "the hidden-state capture as its own copy (instead of fused into the boundary)",
    "GQAMIX": "the in-tree block-sparse decode for the generation rows of mixed steps",
    "H2D_CHANGED_SPAN": "the whole packed 2-D upload (instead of each row's changed column span)",
    "HOSTTAIL": (
        "the stock host step loop: turns off HOST_D2H_SIDE, SMALL_H2D, BATCH_CHANGED_ONLY, "
        "SKIP_ACCEPTED_UPLOAD, PHILOX_PACK and KV_LENS_ONCE together"
    ),
    "HOST_D2H_SIDE": "the stock device-to-host copies on the main stream",
    "INDEX_EARLY": "the index scorer without its early launch at one request",
    "KV_LENS_ONCE": "both KV-length passes of the step",
    "LC8_SYNC3": "the long-context decode's original barrier sequence",
    "MARKOV_CHAIN_BATCH1_KERNEL": "the Markov chain's general kernel for batch-1 calls",
    "MARKOV_LL": "the batch-1 Markov kernel's flag-word exchange",
    "MCAG": "the multicast all-gather replaced by the NCCL all-gather",
    "MOE_B1WIDE": "the MoE engine's batch-1 config at the wide token counts",
    "MOE_LGREPL16": "the MoE engine without its 16-token gather replication",
    "MOE_M8CFG": "the MoE engine's generic config at the 8-token rows",
    "MOE_TAIL_TRIGGER": "the MoE engine's early all-reduce release",
    "MSA_PROXY_PREBUILD": "compiling each MSA proxy fmha_sm100 variant on its first call (instead of before warmup)",
    "PACK_EARLY_COPY": "the step prologue copying the pinned window itself",
    "PACK_IN_GRAPH": "the eager step-input upload and scatter between the graphs",
    "PACK_PSCATTER": "the serial one-CTA step-input scatter (and no in-graph pack)",
    "PHILOX_PACK": "the stock separate Philox uploads",
    "PLANW": "the MoE engine plan's equal FC2 weight",
    "PRESTAGE_OLD": "the index scorer loading every index-K tile after the grid wait",
    "PRODUCER_EARLY_TRIGGER_SHORTFIX": "the producer's early trigger at every sequence length",
    "QKVPROD": "the QKV GEMV followed by the producer",
    "R1": "the 48-row decode step bound (instead of 64)",
    "REJ_MC": "the stock trtllm::chain_speculative_sampling",
    "S4_FP32SUM": "the Markov chain's bf16-rounded base + bias sums",
    "S4_LOGITS": "the in-tree chain under advanced draft sampling",
    "SCORE_SET": "the index scorer publishing its scores at its exit",
    "SELECT_DSL": "the built-in long-row select",
    "SHARED_SWIGLU_MXFP8": "the bf16 SwiGLU + quantize for the shared expert's down projection",
    "SKIP_ACCEPTED_UPLOAD": "the stock per-step upload of the accepted-draft counts",
    "SLOT_IDS_CHANGED": "the per-step upload of the drafter slot ids",
    "SMALL_H2D": "the stock per-step uploads",
    "SOFTMAX_MC": "the stock FlashInfer softmax",
    "STEP_GATHER_FUSED": "the engine's eager previous-batch input gathers at the step boundary",
    "STRICT_ACCEPT_FUSED": "the in-tree strict acceptance and next-step input tokens",
    "TAIL_BOOKKEEP": "every in-tree line of the decode tail bookkeeping",
    "TAIL_BOOKKEEP_SPEC": "the Markov chain's stock KV rewind and biased-logits glue",
    "TOPPDET_T3": "the deterministic top-p renorm without its candidate-list radix passes",
    "TOPPDET_V3": "the deterministic top-p renorm's v2 kernel",
    "TOPP_DET": "the stock FlashInfer top_p_renorm_probs",
    "TSEL": "the built-in prefill tile selector",
    "VERIFY_LOGITS_ALLGATHER_MC": "the NCCL all-gather + strided copy of the verify logits",
    "VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN": "the NCCL verify-logits all-gather at draft lengths other than 3",
    "VERIFY_LOGITS_COPY_FUSED": "the gather_ids copy, vocab concat and fp32 cast of the verify logits",
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


def _read_disabled() -> FrozenSet[str]:
    """The registered ids switched off in this process; logs each unregistered switch variable once."""
    out = set()
    for name, value in os.environ.items():
        if not name.startswith(_PREFIX):
            continue
        item_id = name[len(_PREFIX) :]
        if item_id not in ITEMS:
            logger.warning(f"{name} is not a switch of this MiniMax-M3.1 target; ignored")
            continue
        if _is_set(name, value):
            out.add(item_id)
    return frozenset(out)


# The item ids switched off in this process, read once at import.
_DISABLED: FrozenSet[str] = _read_disabled()


def m31_disabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_DISABLE_<item_id>`` turned the item off for this process."""
    if item_id not in ITEMS:
        raise KeyError(
            f"{item_id!r} is not a registered switch of this target (m31_switches.ITEMS)"
        )
    return item_id in _DISABLED
