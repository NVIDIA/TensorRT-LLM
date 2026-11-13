# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kill switches of this MiniMax-M3.1 target's default-on items (each target keeps its own copy: targets share
only the catalog).

Every item is on by default; ``TRTLLM_M31_DISABLE_<ID>=1`` turns item ``<ID>`` off and restores the path it replaced.
The environment is read once, at import, so every rank and every step of a process sees the same setting (a step must
not change its graph mid-run). ``ITEMS`` registers every id this target reads, with what its switch restores;
``m31_disabled`` raises on an id that is not registered. A few items are off by default (``OPT_INS``):
``TRTLLM_M31_ENABLE_<ID>=1`` turns one on and its ``TRTLLM_M31_DISABLE_<ID>=1`` still wins; ``m31_enabled`` raises on
an id that is not registered there.

Values (case-insensitive, surrounding blanks ignored): ``1`` / ``true`` / ``on`` / ``yes`` set a switch; empty / ``0`` /
``false`` / ``off`` / ``no`` leave it unset; anything else raises at import, naming the variable, so a typo fails loudly
instead of silently turning an item on or off. A ``TRTLLM_M31_DISABLE_*`` variable naming an id this target does not
register (another target's switch in a shared environment, or a renamed id) is logged once and otherwise ignored.
"""

import os
from typing import Dict, FrozenSet

from tensorrt_llm.logger import logger

_PREFIX = "TRTLLM_M31_DISABLE_"
_ENABLE_PREFIX = "TRTLLM_M31_ENABLE_"
_SET = frozenset(("1", "true", "on", "yes"))
_UNSET = frozenset(("", "0", "false", "off", "no"))

# Item id -> what TRTLLM_M31_DISABLE_<id>=1 restores.
ITEMS: Dict[str, str] = {
    "ADV_PAR": "the serial drafter context advance (instead of the parallel advance + read view)",
    "BATCH_CHANGED_ONLY": "the every-step copies of the drafter's batch indices and batch-to-slot map",
    "BLOCKOFF_GUARD": "the drafter KV cache manager's block-table copy on every step",
    "BLOCKOFF_PATCH": "the full-width block-table copy on steps that change one page index per row",
    "CTXNORMQ": "the fused all-reduce + norm and the consumers' own quantize at large prefill layer boundaries",
    "DHEAD_MX4": "the BF16 draft head GEMM (no MXFP8 copy of this rank's head shard)",
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
    "DRAFT_KV_REWIND_FUSED": "the in-tree draft-forward KV rewind bookkeeping",
    "DRAFT_S4": "the in-tree Markov chain (instead of the one-launch sharded chain)",
    "EX11": "flashinfer's three launches for the drafter q / k RMSNorm + partial RoPE",
    "FC_SHARD": "the drafter's replicated fc (instead of an output-column shard + all-gather)",
    "FRONT_CAPFORK": "the hidden-state capture copy in line on the main stream",
    "GEMV_PUSHOPT": "the MXFP8 GEMV's earlier push epilogue",
    "GQAMIX": "the in-tree block-sparse decode for the generation rows of mixed steps",
    "LC_ONEDIV": "the long-context decode without its one-division normalization",
    "MARKOV_CHAIN_BATCH1_KERNEL": "the Markov chain's general kernel for batch-1 calls",
    "MARKOV_GRID": "the batch-1 Markov kernel on the family grid (not this target's wider grid)",
    "MARKOV_LL": "the batch-1 Markov kernel's flag-word exchange",
    "MEMCPY4B": "the drafter's seq_lens copy on the main stream (not the fork's stream)",
    "MEMCPY86K": "the drafter's own copy of its residual stream (instead of starting it in the noise block)",
    "MOE_HANDOFF": "the MoE engine's FC1 -> FC2 hand-off through the B image at its kernel's default lag",
    "MOE_K128": "the MoE engine without the packaged K128 instruction scheduling",
    "MOE_OUT_SKIP": "the MoE engine's bf16 out store on pushing calls",
    "MOE_SFA_BULK": "the MoE engine's per-stage shared FC1 scale copies",
    "MOE_TAIL_TRIGGER": "the MoE engine's early all-reduce release",
    "MSA_PROXY_PREBUILD": "compiling each MSA proxy fmha_sm100 variant on its first call (instead of before warmup)",
    "ONEDIV_CAP": "the one-division normalization at every row count",
    "OPROJ_TMEM256": "every o_proj GEMV tile in TMEM (no 256-column allocation)",
    "PACK_CPP_SCATTER": "the Triton parallel scatter on eager step-input flushes",
    "PADFOLD": "the draft head GEMM into a fresh tensor and a copy into the padded logits buffer",
    "PREFILL_AR_MNNVL": "AllReduce(AUTO) at the prefill and mixed-step layer boundaries",
    "QKV_FRONT": "the boundary all-reduce, QKV GEMV and producer as three launches",
    "QKV_PRODFOLD": "the QKV GEMV and the separate producer launch",
    "QKV_PRODFOLD_RING2": "the folded QKV GEMV at its default ring depth",
    "R6": "the BF16 hand-off at the dense-MLP layer boundaries (the next QKV quantizes its own input)",
    "REJ_MC": "the stock trtllm::chain_speculative_sampling",
    "S10_K7": "the lean drafter glue up to draft length 3 only",
    "S2": "the drafter's replicated fc (no row-parallel split)",
    "S4_FP32SUM": "the Markov chain's bf16-rounded base + bias sums",
    "S4_LOGITS": "the in-tree chain under advanced draft sampling",
    "S5": "the drafter's sliding-window decode at the full max_seq_len (no window cap)",
    "S9BVR": "AllReduce(AUTO) for the drafter's row-parallel all-reduces at every row count",
    "SELECT_EARLY": "the long-row select's sparse-decode trigger at the select's exit",
    "SILU_VEC": "the stock Triton silu_and_mul everywhere",
    "SKIP_ACCEPTED_UPLOAD": "the stock per-step upload of the accepted-draft counts",
    "SM107_DRAFT_FMHA_SM100F": "the drafter attention on thop.attention's in-tree runner",
    "SOFTMAX_MC": "the stock FlashInfer softmax",
    "STEP_GATHER_FUSED": "the engine's eager previous-batch input gathers at the step boundary",
    "STRICT_ACCEPT_FUSED": "the in-tree strict acceptance and next-step input tokens",
    "T2SC_TOKEN_MAJOR": "the block-major selector score view",
    "TAILBK_F": "the k_norm weight multiply as its own launch before the fused context store",
    "TAILGLUE_A": "the fused context store, advance and forward indices as separate launches",
    "TAILGLUE_B": "the two launches of next_new_tokens and the forward-tail fold",
    "TAIL_BOOKKEEP": "every in-tree line of the decode tail bookkeeping",
    "TAIL_BOOKKEEP_SPEC": "the Markov chain's stock KV rewind and biased-logits glue",
    "TAIL_DEFER_CAPTURE": "the capture copies on the capture stream (not kept by reference for the fork)",
    "TAIL_DSFOLD_FWD": "the step gather only (no store at the forward's tail)",
    "TAIL_FC_GATHER": "the NCCL all-gather + concat of the fc shards inside the fork",
    "TAIL_FORCEVEC": "the in-tree forced-acceptance draw",
    "TAIL_FORK": "the stock order of the drafter's input work (no aux-stream fork)",
    "TAIL_FORK_KV": "the context K / V GEMM + k_norm inside the store",
    "TAIL_FORK_REFRESH": "the block-table refresh before the sampling",
    "TAIL_LEAN_WORKER": "the worker-side drafter glue without the cached arange tensors and the identity gather",
    "TOPPDET_K8": "the deterministic top-p renorm without its wide histogram",
    "TOPPDET_T3": "the deterministic top-p renorm without its candidate-list radix passes",
    "TOPPDET_V3": "the deterministic top-p renorm's v2 kernel",
    "TOPPDET_V4": "the deterministic top-p renorm without its list path",
    "TOPP_DET": "the stock FlashInfer top_p_renorm_probs",
    "TP4WC_B2LAG": "the MoE engine's lag-2 schedule with one FC2 B buffer",
    "TP4WC_EARLY_TRIG": "the TP4 MoE engine's dependent release at its end (no teardown release)",
    "TP4WC_HOIST": "the TP4 MoE engine / m3_moe path (no hoisted MoE-boundary block)",
    "TP4_ANCESTRAL": "the S4 route through _refine_block_logits (no in-tree ancestral draft)",
    "TP4_FSEL": "the CuTe DSL long-row select (no fused top-16 select)",
    "TP4_GUMBEL_DRAFT": "the per-position ancestral draft loop (no Gumbel block sampler)",
    "TP4_MOE_ENGINE": "the m3_moe engine at the 8-row decode step",
    "TP4_WHOLECHAIN": "the m3_moe path: turns off TP4_MOE_ENGINE and TP4WC_HOIST together",
    "TP8WIDE": "every TP8 wide-row item (TP8WIDE_SOFTMAX)",
    "TP8WIDE_SOFTMAX": "the multi-CTA softmax's 8-row cap at 16 write CTAs",
    "VERIFY_LOGITS_ALLGATHER_MC": "the NCCL all-gather + strided copy of the verify logits",
    "VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN": "the NCCL verify-logits all-gather at draft lengths other than 3",
    "VERIFY_LOGITS_COPY_FUSED": "the gather_ids copy, vocab concat and fp32 cast of the verify logits",
    "VR_BANDS": "the 32-row decode step bound",
}

# Default-off item id -> what TRTLLM_M31_ENABLE_<id>=1 turns on. The item's TRTLLM_M31_DISABLE_<id>=1 still wins.
OPT_INS: Dict[str, str] = {
    "AR_AUTOTUNE": "leaves the AllReduce autotune on (the target turns it off by default for run-to-run determinism)",
    "DHEAD_MX4": "accepted; the MXFP8 draft head is on by default",
    "DRAFTER_MXFP8": "the drafter's projections on the MXFP8 GEMV (drafter-only precision)",
    "DRAFT_AR_NORM_FUSED": "the drafter's all-reduce fused with the add + RMSNorm after it (not bitwise)",
    "MARKOV_GRID": "the batch-1 Markov kernel on this target's wider grid",
    "MEMCPY4B": "the drafter's seq_lens copy on the fork's stream",
    "QKV_PRODFOLD_RING2": "the folded QKV GEMV at ring depth 2",
    "R6": "the dense-MLP layer boundary publishing the next layer's MXFP8 QKV activation (not bitwise)",
    "S2": "the drafter's fc split row-parallel with one fp32 all-reduce (not bitwise)",
    "S5": "accepted; the window-capped drafter decode is on by default here",
    "SELECT_K13": "the 13-key long-row select variant",
    "SELECT_W512": "the 512-thread long-row select variant",
    "T2SC_QOWN": "the index scorer's q-owner schedule",
    "TAIL_DEFER_CAPTURE": "the capture layers' residuals kept by reference and copied inside the fork",
    "TP4_ANCESTRAL": "the in-tree DSpark ancestral draft (instead of the S4 route)",
    "TP4_GUMBEL_DRAFT": "the family Gumbel-max block sampler for the ancestral draft",
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


def _read(prefix: str, registry: Dict[str, str]) -> FrozenSet[str]:
    """The ids of ``registry`` set with ``prefix`` in this process; logs each unregistered variable once."""
    out = set()
    for name, value in os.environ.items():
        if not name.startswith(prefix):
            continue
        item_id = name[len(prefix) :]
        if item_id not in registry:
            logger.warning(f"{name} is not a switch of this MiniMax-M3.1 target; ignored")
            continue
        if _is_set(name, value):
            out.add(item_id)
    return frozenset(out)


# The item ids switched off in this process, read once at import.
_DISABLED: FrozenSet[str] = _read(_PREFIX, ITEMS)
# The opt-in ids switched on in this process, read once at import.
_ENABLED: FrozenSet[str] = _read(_ENABLE_PREFIX, OPT_INS)


def m31_disabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_DISABLE_<item_id>`` turned the item off for this process."""
    if item_id not in ITEMS:
        raise KeyError(
            f"{item_id!r} is not a registered switch of this target (m31_switches.ITEMS)"
        )
    return item_id in _DISABLED


def m31_enabled(item_id: str) -> bool:
    """Whether ``TRTLLM_M31_ENABLE_<item_id>`` opted the default-off item in for this process."""
    if item_id not in OPT_INS:
        raise KeyError(
            f"{item_id!r} is not a registered opt-in of this target (m31_switches.OPT_INS)"
        )
    return item_id in _ENABLED
