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
    "AGG_PREFILL_CUBLASLT": "the stock MXFP8 GEMM for the prefill projections the pinned cuBLASLt GEMM takes",
    "AGG_PROXY_Q128": "the planner's tile for the prefill indexer proxy (instead of tile 128)",
    "AGG_PROXY_SPLITCAP5": "a split cap of 16 for the 16K-row prefill indexer proxy",
    "AGG_PROXY_SPLITS": "the prefill indexer proxy without its row / KV-size split ladder",
    "AGG_SP": "the ordinary prefill layer boundaries (same as TP2AGG_SP)",
    "AGG_SP_PREMOE": "the gathered residual before the pre-MoE boundary (sequence parallel after the MoE only)",
    "AGG_TSEL": "the built-in prefill tile selector",
    "BATCH_CHANGED_ONLY": "the every-step copies of the drafter's batch indices and batch-to-slot map",
    "BLOCKOFF_GUARD": "the drafter KV cache manager's block-table copy on every step",
    "BLOCKOFF_PATCH": "the full-width block-table copy on steps that change one page index per row",
    "CTXNORMQ": "the fused all-reduce + norm and the consumers' own quantize at large prefill layer boundaries",
    "DECODE_LAYER_WIDE_ROWS": "the 8-row decode step bound (every wider step runs the generic layers)",
    "DRAFTER_BLOCK_TABLE_REFRESH_FUSED": "the drafter's unfused block-table refresh",
    "DRAFTER_BLOCK_TABLE_REFRESH_GRID": (
        "the serial block-table refresh kernel (instead of the row x column-chunk grid)"
    ),
    "DRAFTER_CTX_ADVANCE_FUSED": "the drafter's unfused context advance",
    "DRAFTER_CTX_KV_STORE_FUSED": "the drafter's unfused context KV store",
    "DRAFTER_FORWARD_INDICES_FUSED": "the drafter's unfused forward indices",
    "DRAFTER_FUSED_GLUE": "every fused drafter glue launch (DRAFTER_CTX_KV_STORE_FUSED..E)",
    "DRAFTER_HISLAYOUT": "the drafter projections without the layout routing (the stock GEMMs)",
    "DRAFTER_HISLAYOUT_WIDE": "the 7-row bound of the drafter layout routing (the stock BF16 GEMM at 14..28 rows)",
    "DRAFTER_INPUTS_FUSED": "the drafter's unfused input preparation",
    "DRAFTER_LEAN_GLUE": "the drafter glue without the strided views",
    "DRAFTER_MXFP8": "the drafter's BF16 GEMMs (instead of the MXFP8 GEMV)",
    "DRAFT_A1": "the stock draft-head gather (instead of the prebuilt verify-logits workspace)",
    "DRAFT_AG_BF16": "the stock BF16 draft head and Markov consumer for the gathered draft rows",
    "DRAFT_APPEND_FUSE": (
        "the drafter's per-layer paged K / V append (instead of writing it in the q / k norm + RoPE launch)"
    ),
    "DRAFT_AR_NORM_CLUSTER": "the drafter's stock fused all-reduce + norm at every row count",
    "DRAFT_AR_NORM_CLUSTER_WIDE": "the 8-row bound of the drafter's multicast all-reduce + norm",
    "DRAFT_AR_NORM_FUSED": "the drafter's stock all-reduce followed by its add + RMSNorm",
    "DRAFT_HEAD_NO_SLICE": "the contiguous slice copy of a draft head copy's logits",
    "DRAFT_KV_REWIND_FUSED": "the in-tree draft-forward KV rewind bookkeeping",
    "DRAFT_LOWM_GEMM_HANDOFF": (
        "DRAFT_LOWM_GEMM leaving every projection the MXFP8 GEMV routed alone (with DRAFT_LOWM_GEMM on)"
    ),
    "DRAFT_PROB1": "the stock per-position ancestral draft loop",
    "DRAFT_S4": "the in-tree Markov chain (instead of the one-launch sharded chain)",
    "DRAFT_SWIGLU": "the drafter MLPs' stock activation (instead of the shared native SwiGLU)",
    "EMBED16": "the generic embedding at 9..16 rows",
    "EX11": "flashinfer's three launches for the drafter q / k RMSNorm + partial RoPE",
    "FRONT_CAPFORK": "the hidden-state capture copy in line on the main stream",
    "FRONT_CAPFORK_WIDE": "the in-line capture copy past the batch-1 verify rows",
    "FSEL_WIDE": "the DSL select at 9..32 query rows (the FSEL top-16 select up to 8 rows)",
    "GEMV_PUSHOPT": "the MXFP8 GEMV's earlier push epilogue",
    "GEMV_PUSHOPT16": "the MXFP8 GEMV's earlier push epilogue at the 9..16-row pushes",
    "GEN_GLUE_NATIVE": "the drafter's stock SwiGLU and SiLU and the Triton step-input scatter",
    "GEN_NATIVE_GLUE": "every native glue part: TP2_FSEL, GEN_GLUE_NATIVE and SHARED_SWIGLU_MXFP8_NATIVE together",
    "GQAMIX": "the in-tree block-sparse decode for the generation rows of mixed steps",
    "H2D_CHANGED_SPAN": "the whole packed 2-D upload (instead of each row's changed column span)",
    "HOSTTAIL": (
        "the stock host step loop: turns off HOST_D2H_SIDE, SMALL_H2D, BATCH_CHANGED_ONLY, SKIP_ACCEPTED_UPLOAD, "
        "PHILOX_PACK and KV_LENS_ONCE together"
    ),
    "HOST_CAPTURE_TAIL": (
        "the in-line capture copies and the in-tree tail bookkeeping (FRONT_CAPFORK and TAIL_BOOKKEEP together)"
    ),
    "HOST_D2H_SIDE": "the stock device-to-host copies on the main stream",
    "INDEX_SCORE_T2SC_WIDE": "the sm_103-derived index scorer at 3 and 4 requests",
    "KV_LENS_ONCE": "both KV-length passes of the step",
    "LC2_ALLROWS": "the long-context decode only inside its row band",
    "LC2_IDENTITY": "the Triton route for identity-selection steps",
    "LC2_MIXED": "the Triton sparse decode for the generation rows of mixed steps",
    "LC_ONEDIV": "the long-context decode without its one-division normalization",
    "LINEAR_CUBLASLT": "FlashInfer's MXFP8 GEMM for the 33..64-row projections cuBLASLt takes",
    "MARKOV16": "the catalog Markov chain at every batch",
    "MARKOV16_B1": "the catalog's batch-1 Markov kernel at batch 1",
    "MARKOV16_FOLD": "the Markov chain's pad copy and padded buffer",
    "MARKOV16_W2REGS": "the Markov chain's streamed blocks from global memory at every position",
    "MARKOV_CHAIN_BATCH1_KERNEL": "the Markov chain's general kernel for batch-1 calls",
    "MARKOV_CHAIN_UP_TO_12_REQUESTS": "the 8-request bound of the one-launch Markov chain",
    "MEMCPY86K": "the drafter's own copy of its residual stream (instead of starting it in the noise block)",
    "MOE_EXCHANGE_16_REPLICAS": "the MoE engine exchange's 32 replicas at 7 and 8 tokens",
    "MOE_HALFPLAN": "the MoE engine's split units extended to their end",
    "MOE_HANDOFF": "the MoE engine's FC1 -> FC2 hand-off through the B image at its kernel's default lag",
    "MOE_K128": "the MoE engine without the packaged K128 instruction scheduling",
    "MOE_M8CFG": "the MoE engine's generic config at the 8-token rows",
    "MOE_SFA_BULK": "the MoE engine's per-stage shared FC1 scale copies",
    "MSA_PROXY_PREBUILD": "compiling each MSA proxy fmha_sm100 variant on its first call (instead of before warmup)",
    "MXGEMV_SM107_CFG": "the MXFP8 GEMV's table ring for the SwiGLU-prologue calls",
    "NATIVE_FINALIZE": "the Triton MoE finalize",
    "NATIVE_SWIGLU": "the Triton SwiGLU-OAI activation",
    "NVFP4_STEP_CACHE": "the NVFP4 decode reader's history scheduler built per sparse layer",
    "ONEDIV_CAP": "the one-division normalization at every row count",
    "OPROJ_PUSH_GEMM": "the stock o_proj GEMM and attention boundary all-reduce (instead of the Lamport-push GEMM)",
    "PACK_CPP_SCATTER": "the Triton parallel scatter on eager step-input flushes",
    "PACK_EARLY_COPY": "the step prologue copying the pinned window itself",
    "PACK_IN_GRAPH": "the eager step-input upload and scatter between the graphs",
    "PACK_PSCATTER": "the serial one-CTA step-input scatter (and no in-graph pack)",
    "PHILOX_PACK": "the stock separate Philox uploads",
    "PREFILL_AR_MNNVL": "AllReduce(AUTO) at the prefill and mixed-step layer boundaries",
    "PREFILL_BRK": "no breakable prefill CUDA graph (the prefill target raises under one)",
    "PREFILL_OUT_CLONE": "the prefill hidden state returned without its copy on the current stream",
    "REJ_MC": "the stock trtllm::chain_speculative_sampling",
    "REJ_MC_WIDE": "the multi-CTA rejection sampler's narrow row bound",
    "REJ_SLOTS": "the sampler's gather of the draft distributions before the multi-CTA rejection",
    "REJ_SLOTS_WIDE": "the draft-probability gather before the rejection at the wide row counts",
    "ROUTER_TF32_48_ROWS": "the fp32 router for the 9..48-row steps",
    "S1B_FOLD": "the stock router + routing above the tensor-core router's rows",
    "S1B_FOLD_PDL": "the routing fold launched without programmatic dependent launch",
    "S4FE": "the in-tree elementwise tail of the Markov chain",
    "S4_FP32SUM": "the Markov chain's bf16-rounded base + bias sums",
    "S4_LOGITS": "the in-tree chain under advanced draft sampling",
    "S9BVR": "AllReduce(AUTO) for the drafter's row-parallel all-reduces at every row count",
    "SHARED_SWIGLU_MXFP8": "the bf16 SwiGLU + quantize for the shared expert's down projection",
    "SHARED_SWIGLU_MXFP8_NATIVE": "the Triton producer of the shared expert's MXFP8 SwiGLU",
    "SKIP_ACCEPTED_UPLOAD": "the stock per-step upload of the accepted-draft counts",
    "SM107_DRAFT_FMHA_SM100F": "the drafter attention on thop.attention's in-tree runner",
    "SMALL_DECODE_MOE_ENGINE": "the original TP2 MoE engine at every row count",
    "SMALL_H2D": "the stock per-step uploads",
    "SOFTMAX_MC": "the stock FlashInfer softmax",
    "SOFTMAX_MC_128": "the multi-CTA softmax's 112-row workspace and dispatch bound",
    "SOFTMAX_MC_WIDE": "the multi-CTA softmax's narrow row bound",
    "SOFTMAX_PATHB": "FlashInfer's softmax at 129..448 rows",
    "STEP_GATHER_FUSED": "the engine's eager previous-batch input gathers at the step boundary",
    "STRICT_ACCEPT_FUSED": "the in-tree strict acceptance and next-step input tokens",
    "TAILGLUE_B": "the two launches of next_new_tokens and the forward-tail fold",
    "TAIL_BK3": "the stock post-layer bookkeeping launches",
    "TAIL_BOOKKEEP": "every in-tree line of the decode tail bookkeeping",
    "TAIL_BOOKKEEP_SPEC": "the Markov chain's stock KV rewind and biased-logits glue",
    "TAIL_DRAFT_PROBS_SLOT_COPY": "the stock index_put of the block draft probabilities",
    "TAIL_DSFOLD_FWD": "the step gather only (no store at the forward's tail)",
    "TAIL_FC_GATHER": (
        "the replicated fc in the TAIL_FORK fork (instead of the shard + multicast all-gather, with DRAFT_FC_SHARD on)"
    ),
    "TAIL_FORCEVEC": "the in-tree forced-acceptance draw",
    "TAIL_FORK": "the stock order of the drafter's input work (no aux-stream fork)",
    "TAIL_FORK_B1": "no fork at the batch-1 step's 8 rows",
    "TAIL_FORK_KV": "the context K / V GEMM + k_norm inside the store",
    "TAIL_FORK_REFRESH": "the block-table refresh before the sampling",
    "TAIL_ZERO_PAD_ROWS": "the stock handling of the CUDA-graph padding rows",
    "TOPPDET_K8": "the deterministic top-p renorm without its wide histogram",
    "TOPPDET_T3": "the deterministic top-p renorm without its candidate-list radix passes",
    "TOPPDET_V3": "the deterministic top-p renorm's v2 kernel",
    "TOPPDET_V4": "the deterministic top-p renorm without its list path",
    "TOPP_DET": "the stock FlashInfer top_p_renorm_probs",
    "TP2AGG_ROUTER_F32": "the prefill router's own fp32 cast of its input",
    "TP2AGG_SP": "the ordinary prefill layer boundaries (no sequence-parallel boundary)",
    "TP2CTX_SP_ADDNORM": "the peer add and the NORMQ launch as two kernels",
    "TP2CTX_SP_OVL": "the owned MXFP8 rows pushed before the fused QKV kernel (with TP2_PREFILL_QKV_FUSED on)",
    "TP2CTX_SP_PMOVL": "the pre-MoE products pushed after the router GEMM",
    "TP2WIDE_COMBINE_MXFP8": "o_proj's own quantize of the sparse prefill output",
    "TP2WIDE_COMMGEMV": (
        "the 8-row bounds of the wide decode items (capture fork, drafter all-reduce + norm, first layer, push, GEMV "
        "configs)"
    ),
    "TP2WIDE_DECODE": "the generic layers past 64 decode rows",
    "TP2WIDE_GEMM": "the GEMV at 33..48 rows",
    "TP2WIDE_GEMVCFG": "the shared MXFP8 GEMV config table at 9..32 rows",
    "TP2WIDE_INPUT9": "the generic first layer at 9..64 decode rows",
    "TP2WIDE_INTREE_SCORER": "each request count's own index scorer above 32 rows",
    "TP2WIDE_LC2": "the Triton sparse decode at every row count",
    "TP2WIDE_LC2_448": "the long-context decode's 128-row bound",
    "TP2WIDE_LC2_SPLIT": "the long-context decode's cluster kernel at every row count",
    "TP2WIDE_PUSH": "the 8-row bound of the o_proj push into the attention boundary",
    "TP2WIDE_SCHED_LPT": "the sparse prefill's default work-item order",
    "TP2WIDE_SPARSE_OUT": "the sparse prefill's copied attention output",
    "TP2W_FP4SLOT_FINALIZE": "the routed experts' own NVFP4 quantize of ordinary prefill rows",
    "TP2W_MC_FINALIZE": "the built-in MoE finalize before the boundary all-reduce",
    "TP2W_MOE_FINALIZE": "the TRTLLM-Gen runner's own finalize",
    "TP2W_MOE_TACTIC_PIN": "the AutoTuner's MoE tactic at the pinned row counts",
    "TP2W_S2_TAIL": "the MoE finalize, boundary all-reduce + norm and QKV quantize as three kernels",
    "TP2_DIE_LOCAL": "the 8-row MoE engine's plain roles and weight placement",
    "TP2_FSEL": "the DSL long-row select (instead of the FSEL top-16 select)",
    "TP2_GEMVCFG": "the shared MXFP8 GEMV configs at this target's tuned shapes",
    "TP2_LC2_SMALL": "the long-context decode's 33-row lower bound at draft length 7",
    "TP2_MC_FINALIZE_SMALL": "the MoE finalize fold's 33-row lower bound",
    "TP2_QKV_PRODUCER_FOLD": "the QKV GEMV followed by the producer",
    "TP_GUMBEL_DRAFT": "the DSpark ancestral block sampler without the Gumbel-max override",
    "VERIFY_LOGITS_ALLGATHER_MC": (
        "the NCCL all-gather + strided copy of the verify logits (instead of one multicast all-gather + cast launch)"
    ),
    "VERIFY_LOGITS_ALLGATHER_MC_448": "the 256-row verify-logits workspace",
    "VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN": "the NCCL verify-logits all-gather at draft lengths other than 3",
    "VERIFY_LOGITS_ALLGATHER_MC_WIDE": "the 64-row verify-logits workspace (the NCCL all-gather above 64 rows)",
    "VERIFY_LOGITS_COPY_FUSED": (
        "the logits processor's gather_ids copy, vocab concat and fp32 cast of the verify logits"
    ),
    "VR_BANDS": "the 32-row decode step bound",
    "VR_GEMV48": "FlashInfer's MXFP8 GEMM for the 33..48-row projections the GEMV takes",
    "WC_EARLY_TRIG": "the 8-row MoE engine's teardown release",
    "WC_MOE_ENGINE": "the m3_moe engine at 8 rows",
    "WHOLECHAIN": "every part of the 8-row MoE engine chain (WC_MOE_ENGINE, TP2_DIE_LOCAL, WC_EARLY_TRIG)",
    "X1": "the routed experts' own quantize of the router's input",
    "CTXNORM_LB2": "the CTX norm + MXFP8 boundary kernel at one CTA per SM",
    "TSEL_NARROW": "the prefill tile select's tile kernel for every tile (no narrow-row kernel)",
    "REJ_V3": "the rejection sampler's v2 kernel on every device",
    "TOPP_V6": "the top-p renorm's staged summation loop on every device",
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
    "DRAFT_FC_SHARD": (
        "the drafter's replicated fc as this rank's column shard + an all-gather at the draft-length verify rows"
    ),
    "DRAFT_FC_TP_SPLIT": "the drafter's replicated fc split row-parallel with one fp32 all-reduce (not bitwise)",
    "DRAFT_FMHA_BATCH2_SPLIT": (
        "the drafter's batch-2 window decode as two per-request window-capped calls (not bitwise)"
    ),
    "DRAFT_HEAD_MXFP8": "the drafter head's shard GEMM on an MXFP8 copy of the shard (not bitwise)",
    "DRAFT_HEAD_NVFP4": (
        "the drafter head's shard GEMM on an NVFP4 copy of the shard (not bitwise; wins over DRAFT_HEAD_MXFP8)"
    ),
    "DRAFT_LOGITS_ALLGATHER_MC": "the rejection sampler's draft-logits all-gather as one multicast launch",
    "DRAFT_LOWM_GEMM": (
        "the drafter's o_proj / down GEMMs the MXFP8 GEMV does not run on FlashInfer's low-M BF16 kernels (not bitwise)"
    ),
    "DRAFT_MLP_MXFP8_14_ROWS": "the drafter MLP's gate_up + SwiGLU at 14 rows as MXFP8 quantize + GEMV (not bitwise)",
    "DRAFT_WINDOW_CAPPED_DECODE": "the window-capped max_seq_len for the drafter's sliding-window decode (not bitwise)",
    "EMBED16_LAYER0_DECODE": (
        "the 9..16-row step's layer 0 on the decode layer (FlashInfer RMSNorm of the gathered rows)"
    ),
    "LC_TP2": "no Triton-route engagement line for the long-context rows the long-context decode leaves (log only)",
    "MOE_ENGINE_9_TO_13_ROWS": (
        "the 9..13-row MoE engine (a TP4-width kernel: at this target's width its build fails and the generic MoE runs)"
    ),
    "MOE_ENGINE_FC2_LAG": "the 8-row MoE engine's FC2 six groups behind FC1 (scheduling only, bitwise)",
    "MOE_ROUTE_IDS_9_TO_16_ROWS": "the 9..16-row generation step's MoE routed from top-4 ids computed by one kernel",
    "R6_DENSE_PREQUANT": "the dense-MLP boundary's MXFP8 QKV hand-off to the next layer (not bitwise)",
    "SLOT_IDS_CHANGED": (
        "the spec sampler's slot-id table uploaded only when it changes (RuntimeHooks spec_slot_ids_changed_only)"
    ),
    "TAIL_DEFER_CAPTURE": (
        "the MC finalize boundary's capture residuals kept by reference and written before the drafter reads them"
    ),
    "TAIL_FORK_AFTER_VERIFY_GATHER": "the TAIL_FORK fork launched after the verify logits' multicast all-gather",
    "TAIL_FORK_ISSUE_AFTER_VERIFY_GATHER": (
        "the TAIL_FORK fork's kernels issued after the verify logits' all-gather launch"
    ),
    "TP2_LC2_SMALL_DRAFT_LEN_3": "the long-context decode from 9 rows at draft length 3 too (not bitwise)",
    "TP2_PREFILL_QKV_FUSED": (
        "the large prefill steps' sparse-layer QKV + index MXFP8 GEMM with the producer in its epilogue (not bitwise)"
    ),
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
