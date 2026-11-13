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
    "ADV_PAR": "the serial drafter context advance (instead of the parallel advance + read view)",
    "BATCH_CHANGED_ONLY": "the every-step copies of the drafter's batch indices and batch-to-slot map",
    "BLOCKOFF_GUARD": "the drafter KV cache manager's block-table copy on every step",
    "BLOCKOFF_PATCH": "the full-width block-table copy on steps that change one page index per row",
    "CTXNORMQ": "the fused all-reduce + norm and the consumers' own quantize at large prefill layer boundaries",
    "DECODE_LAYER_WIDE_ROWS": "the 8-row decode step bound (every wider step runs the generic layers)",
    "DRAFTER_BLOCK_TABLE_REFRESH_FUSED": "the drafter's unfused block-table refresh",
    "DRAFTER_BLOCK_TABLE_REFRESH_GRID": "the serial block-table refresh kernel (not the row x column-chunk grid)",
    "DRAFTER_CTX_ADVANCE_FUSED": "the drafter's unfused context advance",
    "DRAFTER_CTX_KV_STORE_FUSED": "the drafter's unfused context KV store",
    "DRAFTER_FORWARD_INDICES_FUSED": "the drafter's unfused forward indices",
    "DRAFTER_FUSED_GLUE": "every fused drafter glue launch (DRAFTER_*_FUSED, DRAFTER_BLOCK_TABLE_REFRESH_GRID)",
    "DRAFTER_HISLAYOUT": "the drafter projections without the TP8 layout routing",
    "DRAFTER_INPUTS_FUSED": "the drafter's unfused input preparation",
    "DRAFTER_LEAN_GLUE": "the drafter glue without the strided views",
    "DRAFTER_MXFP8": "the drafter's BF16 GEMMs (instead of the MXFP8 GEMV)",
    "DRAFT_APPEND_FUSE": "the drafter's per-layer paged K / V append (instead of writing it in the q / k norm + RoPE "
    "launch)",
    "DRAFT_AR_NORM_FUSED": "the drafter's stock all-reduce followed by its add + RMSNorm",
    "DRAFT_ATTN_PDL": "the drafter attention without programmatic dependent launch",
    "DRAFT_GEMV_RING_RETUNE": "the drafter GEMV ring depths QKV 6 and o 12",
    "DRAFT_GU_SWIGLU_EPI": "the drafter MLP SwiGLU as its own launch (instead of the gate_up GEMV epilogue)",
    "DRAFT_KV_REWIND_FUSED": "the in-tree draft-forward KV rewind bookkeeping",
    "DRAFT_QKV_RING2": "the default ring depth of the drafter QKV GEMV at 7 rows",
    "DRAFT_S4": "the in-tree Markov chain (instead of the one-launch sharded chain)",
    "EX11": "flashinfer's three launches for the drafter q / k RMSNorm + partial RoPE",
    "FC_SHARD": "the drafter's replicated fc (instead of an output-column shard + all-gather)",
    "FOLD_ACCEPT_COUNTS_INT32": "the int64 copy of the accept counts before the fused context store and advance",
    "FOLD_BLOCK_TABLE_IN_INPUTS": "the drafter block-table refresh as its own launch on pure-decode steps",
    "FOLD_DRAFT_HEAD_STRIDED_OUT": "the draft head GEMM's contiguous output + strided copy into the logits buffer",
    "FOLD_FC_SHARD_GATHER_MC": "the NCCL all-gather + concat of the drafter fc shard",
    "FOLD_KV_LOC_TAIL_FILL": "the MSA metadata's own out_cache_loc tail fill",
    "FOLD_NEXT_TOKENS_IN_TAIL": "the separate launch for the strict accept's next new tokens",
    "FOLD_PHILOX_IN_REJECTION": "the separate add for the Philox window of the draft block",
    "FOLD_TOPP_SLOT_WRITE": "the store of the draft block's top-p renorm rows after the sampler",
    "FRONT_CAPFORK": "the hidden-state capture copy in line on the main stream",
    "GQAMIX": "the in-tree block-sparse decode for the generation rows of mixed steps",
    "H2D_CHANGED_SPAN": "the whole packed 2-D upload (instead of each row's changed column span)",
    "HOSTTAIL": "the stock host step loop: turns off HOST_D2H_SIDE, SMALL_H2D, BATCH_CHANGED_ONLY, "
    "SKIP_ACCEPTED_UPLOAD, PHILOX_PACK and KV_LENS_ONCE together",
    "HOST_D2H_SIDE": "the stock device-to-host copies on the main stream",
    "KV_LENS_ONCE": "both KV-length passes of the step",
    "LINEAR_CUBLASLT": "FlashInfer's MXFP8 GEMM for the 33..64-row projections cuBLASLt takes",
    "LOCAL_MOE": "the stock MoE engine (instead of the local-placement variant)",
    "LOCAL_MOE_WS_RECEIPT": "the local MoE engine without its workspace receipt",
    "MARKOV_CHAIN_BATCH1_KERNEL": "the Markov chain's general kernel for batch-1 calls",
    "MARKOV_LL": "the batch-1 Markov kernel's flag-word exchange",
    "MOE_B1SPLIT": "one warp group quantizing every B1 token of the 8-token MoE engine",
    "MOE_EXCHANGE_16_REPLICAS": "the MoE engine exchange's 32 replicas at 7 and 8 tokens",
    "MOE_FC1_GEOMETRY": "the local MoE engine's stock FC1 geometry",
    "MOE_HALFPLAN": "the MoE engine's split units extended to their end",
    "MOE_K128": "the MoE engine without the packaged K128 instruction scheduling",
    "MOE_M8CFG": "the MoE engine's generic config at the 8-token rows",
    "MOE_OUT_SKIP": "the MoE engine's bf16 out store on pushing calls",
    "MOE_SFA_BULK": "the MoE engine's per-stage shared FC1 scale copies",
    "MOE_SH20": "the MoE engine's shared expert on its stock CTA count",
    "MOE_TAIL_TRIGGER": "the MoE engine's early all-reduce release",
    "MOE_XDEFER": "the MoE engine's x copy at its stock place in the 8-token B1 path",
    "MSA_PROXY_PREBUILD": "compiling each MSA proxy fmha_sm100 variant on its first call (instead of before warmup)",
    "MXGEMV_SM107_CFG": "the MXFP8 GEMV's table ring for the SwiGLU-prologue calls",
    "ONEDIV_CAP": "LC8_ONEDIV at every row count (instead of up to 8 query rows)",
    "PACK_CPP_SCATTER": "the Triton parallel scatter on eager step-input flushes",
    "PACK_EARLY_COPY": "the step prologue copying the pinned window itself",
    "PACK_IN_GRAPH": "the eager step-input upload and scatter between the graphs",
    "PACK_PSCATTER": "the serial one-CTA step-input scatter (and no in-graph pack)",
    "PHILOX_PACK": "the stock separate Philox uploads",
    "REJ_MC": "the stock trtllm::chain_speculative_sampling",
    "REJ_SLOTS": "the sampler's gather of the draft distributions before the multi-CTA rejection",
    "ROUTER_TF32_48_ROWS": "the fp32 router for the 9..48-row steps",
    "S4_FP32SUM": "the Markov chain's bf16-rounded base + bias sums",
    "S4_LOGITS": "the in-tree chain under advanced draft sampling",
    "SELECT_EARLY": "the long-row select's sparse-decode trigger at the select's exit",
    "SHARED_SWIGLU_MXFP8": "the bf16 SwiGLU + quantize for the shared expert's down projection",
    "SHARED_SWIGLU_MXFP8_NATIVE": "the Triton producer of the shared expert's MXFP8 SwiGLU",
    "SILU_VEC": "the stock Triton silu_and_mul everywhere",
    "SKIP_ACCEPTED_UPLOAD": "the stock per-step upload of the accepted-draft counts",
    "SM107_DRAFT_FMHA_SM100F": "the drafter attention on thop.attention's in-tree runner",
    "SMALL_H2D": "the stock per-step uploads",
    "SOFTMAX_MC": "the stock FlashInfer softmax",
    "STEP_GATHER_FUSED": "the engine's eager previous-batch input gathers at the step boundary",
    "STRICT_ACCEPT_FUSED": "the in-tree strict acceptance and next-step input tokens",
    "TAIL_BOOKKEEP": "every in-tree line of the decode tail bookkeeping",
    "TAIL_BOOKKEEP_SPEC": "the Markov chain's stock KV rewind and biased-logits glue",
    "TAIL_DRAFT_PROBS_SLOT_COPY": "the stock index_put of the block draft probabilities",
    "TOPPDET_K8": "the deterministic top-p renorm without its wide histogram",
    "TOPPDET_T3": "the deterministic top-p renorm without its candidate-list radix passes",
    "TOPPDET_V3": "the deterministic top-p renorm's v2 kernel",
    "TOPPDET_V4": "the deterministic top-p renorm without its list path",
    "TOPP_DET": "the stock FlashInfer top_p_renorm_probs",
    "TP8WIDE": "every TP8 wide-row item (TP8WIDE_SOFTMAX)",
    "TP8WIDE_SOFTMAX": "the multi-CTA softmax's 8-row cap at 16 write CTAs",
    "TP8_CAUSAL_DRAFT": "the drafter's non-causal in-block window decode",
    "TP8_CTXLEN_FUSE": "the drafter's unfused context-length resume",
    "TP8_DRAFT_HEAD_MXFP8": "the drafter head's BF16 shard GEMM (instead of its MXFP8 copy)",
    "TP8_FSEL": "the built-in top-k select of the sparse decode",
    "TP8_MOETUNE": "the local MoE engine's original config and kernel",
    "VERIFY_LOGITS_ALLGATHER_MC": "the NCCL all-gather + strided copy of the verify logits",
    "VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN": "the NCCL verify-logits all-gather at draft lengths other than 3",
    "VERIFY_LOGITS_COPY_FUSED": "the gather_ids copy, vocab concat and fp32 cast of the verify logits",
    "VR_BANDS": "the 32-row decode step bound",
    "VR_GEMV48": "FlashInfer's MXFP8 GEMM for the 33..48-row projections the GEMV takes",
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


# Default-off items (opt-ins): ``TRTLLM_M31_ENABLE_<ID>=1`` turns item ``<ID>`` on. Each changes the target's numerics
# against the default path (not bitwise) or is not faster on every shape, so it stays off unless asked for. Same strict
# values as the kill switches, read once at import; an opt-in variable naming an id this target does not register is
# logged once and otherwise ignored.
_ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"

# Opt-in id -> what TRTLLM_M31_ENABLE_<id>=1 turns on.
OPT_INS: Dict[str, str] = {
    "AR_AUTOTUNE": (
        "AllReduce AUTO's per-process timing autotuner (the target otherwise pins the C++ size / world heuristic)"
    ),
    "DRAFT_AR_NORM_CLUSTER": (
        "the drafter's fused all-reduce + norm on the multicast all-reduce + norm kernel at 8 rows"
    ),
    "DRAFT_FC_TP_SPLIT": "the drafter's replicated fc split row-parallel with one fp32 all-reduce (not bitwise)",
    "DRAFT_LOGITS_ALLGATHER_MC": "the rejection sampler's draft-logits all-gather as one multicast launch",
    "DRAFT_WINDOW_CAPPED_DECODE": "the window-capped max_seq_len for the drafter's sliding-window decode (not bitwise)",
    "LC8_ONEDIV": "the long-context decode's split-K merge dividing once per output (not bitwise)",
    "R6_DENSE_PREQUANT": "the dense-MLP boundary's MXFP8 QKV hand-off to the next layer (not bitwise)",
    "SLOT_IDS_CHANGED": (
        "the spec sampler's slot-id table uploaded only when it changes (RuntimeHooks spec_slot_ids_changed_only)"
    ),
    "TP8_ANCESTRAL_DRAFT": "the in-tree DSpark ancestral draft under advanced draft sampling",
    "TP8_SHORT_FAMILY": "a separate decode CUDA graph family for short steps (not bitwise)",
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
