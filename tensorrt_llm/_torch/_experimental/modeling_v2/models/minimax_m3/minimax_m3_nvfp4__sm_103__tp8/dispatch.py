# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The decode-step fast paths of this target: the rows each one takes, the catalog entry that certifies them, and
what happens when one cannot run.

One target serves every concurrency with the same code; which fast path a step takes is a pure function of its
shape, written here once (``plan_for``, ``moe_path``, ``router_path``, ``attention_path``, ``linear_path``) and
called by the forward. Every row range is a named constant below, and ``FAST_PATHS`` pairs each with the catalog
entry whose ``CELLS`` certify it: ``test_modeling_v2_m3_dispatch.py`` checks that every row count (or shape key) a
range covers is one of the entry's cells.

Two kinds of not taking a fast path, kept apart:

* **uncovered**: the step is outside every covered range -- prefill, mixed steps, generation wider than the ranges,
  an eager decode step (the GEMV dispatch is a decode-graph one), a long-context request, an Eagle3 capture layer
  that runs the generic layer by design. ``uncovered`` logs the reason once per (path, reason); the caller runs the
  generic path in every mode.
* **missed**: the step is inside a covered range but a precondition is missing -- a workspace, a push descriptor or
  a compiled engine not built, or a kernel declining operands it is certified for. Under ``modeling_v2='require'``
  ``missed`` raises ``FastPathUnavailable`` naming the path, the rows and the precondition; otherwise it logs once
  per (path, reason) and the caller falls back.

Compiled kernels (the MoE engines, the decode GEMVs with their push variants, the index scorer) are built by
``build_targets`` before the executor's warmup; a forward never compiles one, and one it finds missing is a miss.
Other state a fast path needs (multicast workspaces, push descriptors, kernel modules) is built on the eager decode
steps; a CUDA-graph capture only reads it, so a capture that finds it missing is a miss (``capture_missed``) rather
than a silent capture of the fallback.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, FrozenSet, Hashable, Iterable, Mapping, Optional, Tuple

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import OpWrapper
from tensorrt_llm.logger import logger

from .ops.attention.m3_dense_decode import m3_dense_decode
from .ops.attention.m3_fused_score_select import (
    MAX_REQUESTS as FUSED_SCORE_SELECT_KERNEL_MAX_REQUESTS,
)
from .ops.attention.m3_fused_score_select import m3_fused_score_select
from .ops.attention.m3_index_score import m3_index_score
from .ops.attention.m3_producer import m3_producer
from .ops.attention.m3_select_long import MAX_BLOCKS as SELECT_LONG_MAX_BLOCKS
from .ops.attention.m3_select_long import m3_select_long
from .ops.attention.m3_sparse_decode import m3_sparse_decode
from .ops.attention.m3_sparse_decode_lc8 import m3_sparse_decode_lc8
from .ops.comm.m3_mc_allreduce_norm import m3_mc_allreduce_norm
from .ops.comm.m3_mc_finalize_allreduce_norm import m3_mc_finalize_allreduce_norm
from .ops.gemm.m3_draft_lowm import TACTICS as DRAFT_LOWM_TACTICS
from .ops.gemm.m3_draft_lowm import m3_draft_lowm
from .ops.gemm.m3_mx_gemv import DENSE_DOWN, DENSE_GATE_UP, O_PROJ, QKV, QKV_SPARSE, m3_mx_gemv
from .ops.gemm.m3_mx_gemv_bf16 import m3_mx_gemv_bf16
from .ops.moe.m3_moe_engine import m3_moe_engine
from .ops.moe.m3_router_gemv import m3_router_gemv
from .ops.moe.m3_router_gemv_fp4 import m3_router_gemv_fp4
from .ops.moe.m3_router_tc import m3_router_tc
from .ops.norm.m3_embedding_firstnorm import m3_embedding_firstnorm
from .ops.spec.m3_draft_input import m3_draft_input
from .ops.spec.m3_draft_step import MAX_TOKENS as DRAFT_STEP_KERNEL_MAX_TOKENS
from .ops.spec.m3_draft_step import m3_draft_step
from .ops.spec.m3_shard_accept import m3_shard_accept

# --------------------------------------------------------------------------------------------------------------------
# Row ranges
# --------------------------------------------------------------------------------------------------------------------

# The largest generation step the decode target takes: 8 requests verifying 1 + 3 drafts and their padding rows.
DECODE_MAX_TOKENS = 48


@dataclass(frozen=True)
class DecodePlan:
    """The layer-boundary collectives of one decode token bucket (GB300 measurements in the comments)."""

    name: str
    max_tokens: int
    # The attention boundary's multicast all-reduce + residual + RMSNorm (5.05 vs 5.27 us native at 1 token).
    attn_mc_ar: bool
    # The same kernel at the MoE boundary with the shared + routed add folded in: wins up to 48 tokens.
    moe_mc_ar: bool
    # The MoE-boundary all-reduce also emits the next QKV activation in MXFP8 for the decode GEMV / GEMM.
    qkv_prequant: bool


# tiny: every step up to 8 tokens (c1..c8 without speculation, the c1 verify, every draft pass); small: the c2
# verify; verify: the c4 / c8 verifies. Each is one multicast all-reduce bucket (kernels/m3_mc_allreduce.BUCKETS).
DECODE_PLANS = (
    DecodePlan("tiny", 8, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
    DecodePlan("small", 16, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
    DecodePlan("verify", DECODE_MAX_TOKENS, attn_mc_ar=True, moe_mc_ar=True, qkv_prequant=True),
)

# The dense layers' decode attention (trtllm-gen, msa_decode.dense_decode_max_seq_len): the max_seq_len it is handed
# per request count. It only sizes the multi-CTA KV split (every CTA still covers its share of the actual length; the
# result changes only by the split-merge order), and the model maximum over-splits AgentX-length caches at one
# request. Two requests are neutral (the grid is unchanged) and keep the step's bound.
DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH: Mapping[int, int] = MappingProxyType({1: 16384})
# The same cap on the Eagle3 drafter layer's trtllm-gen decode (8 query heads over one KV head per rank): GB300, one
# request, 63.7K cached tokens: 22.5 -> 11.7 us at one query row, 24.3 -> 12.5 us at four; 188K tokens: 28.2 -> 20.2 us.
# Two requests are flat and keep the step's bound.
DRAFT_DECODE_MAX_SEQ_LEN_BY_BATCH: Mapping[int, int] = MappingProxyType({1: 16384})

# The sparse layers' NVFP4 QKV + indexer producer (ops/attention/m3_producer, every step its geometry fits) triggers its
# dependent kernel on entry, so the kernel after it launches and stages its pre-wait loads while the producer runs.
PRODUCER_EARLY_TRIGGER = True

# The fused embedding + first RMSNorm kernel: one owner per row count and index dtype.
EMBED_FIRSTNORM_MAX_TOKENS = 8
# The o_proj pushes its rows into the attention boundary's multicast all-reduce (prepushed mode 2) up to this many
# tokens; wider steps run the GEMV and a plain multicast all-reduce.
ATTN_PUSH_MAX_TOKENS = 16
AR_PUSH_MODE = 2
# The 1..8-token MoE engine (router, routed experts at the TP8 width 384 and the shared expert in one launch) pushes
# its sum into the MoE boundary's all-reduce (prepushed mode 2).
MOE_ENGINE_MAX_TOKENS = 8
MOE_PUSH_MAX_TOKENS = MOE_ENGINE_MAX_TOKENS
# From this many tokens the MoE runs unfinalized through the TRTLLM-Gen runner and its finalize (top-k gather,
# routing weights, shared expert add) runs inside the boundary all-reduce.
MC_FINALIZE_MIN_TOKENS = MOE_ENGINE_MAX_TOKENS + 1
# The router: the fp32 GEMV at one row, the tensor-core router (fused with the routed experts' NVFP4 activation
# quantization) at 2..48 rows.
ROUTER_GEMV_MAX_TOKENS = 1
ROUTER_TC_MAX_TOKENS = 48
# A greedy Eagle3 verify step of up to this many requests verifies with the fused shard argmax + acceptance kernel
# over the target lm_head's BF16 shard logits (kernels/m3_shard_accept), at the configured draft length. The lm_head
# is the in-tree BF16 GEMM: at the TP8 shard (25008 rows) a coded copy of the head saves no time.
SHARD_ACCEPT_DRAFT_LEN = 3
SHARD_ACCEPT_MAX_REQUESTS = DECODE_MAX_TOKENS // (SHARD_ACCEPT_DRAFT_LEN + 1)
# On a fused-verify step of up to this many verify rows the drafter's first fc is forked onto the aux stream right
# after the last captured target layer (spec_worker). At TP8 it forks beside the verify head instead: forked early its
# GEMM lands beside the target tail it was meant to hide under.
FC_EARLY_MAX_ROWS = 0
# A decode-only Eagle3 step of up to this many requests drafts with the fused draft loop (spec_worker): per pass the
# draft input in one launch (kernels/m3_spec_tail.draft_input), the drafter layer's in-tree attention, MLP and
# all-reduces, and one kernel for the BF16 draft head, the cross-rank argmax and the pass's bookkeeping
# (kernels/m3_draft_step), at the configured draft length. The head kernel takes one row per request, at most 8, and
# streams the head's BF16 weight fragments.
DRAFT_STEP_DRAFT_LEN = SHARD_ACCEPT_DRAFT_LEN
DRAFT_STEP_MAX_REQUESTS = DRAFT_STEP_KERNEL_MAX_TOKENS
# The identity-selection sparse decode moves from the per-row producer to the multi-query producer at this many rows
# (kernels/triton_sparse_decode.MQ_MIN_TOKENS).
SPARSE_DECODE_MQ_MIN_TOKENS = 17
# Decode spans of at most this many requests score and select their top-16 blocks in one launch
# (ops/attention/m3_fused_score_select), which loads the older index-K pages before its grid wait; larger spans run
# the index scorer and the long-row selector.
FUSED_SCORE_SELECT_MAX_REQUESTS = FUSED_SCORE_SELECT_KERNEL_MAX_REQUESTS
# The long-row top-16 selector takes engines of 129..LONG_SELECT_MAX_BLOCKS KV blocks of 128 tokens; it holds 2, 4 or
# 9 keys per thread by the row's valid-block count. At most 128 blocks the built-in selector runs.
LONG_SELECT_MIN_BLOCKS = 129
LONG_SELECT_MAX_BLOCKS = SELECT_LONG_MAX_BLOCKS
LONG_SELECT_KEY_WIDTHS = (2, 4, 9)
# The long-row select releases the sparse decode after it right after its own grid wait up to this many query rows;
# wider steps keep the trigger at the select's exit (the early-resident attention CTAs crowd the select grid).
SELECT_EARLY_MAX_ROWS = 8
# The 8-head long-context sparse decode (ops/attention/m3_sparse_decode_lc8) stages the request length, its query rows
# and the request's block-table row before its grid wait up to this many query rows; above, the staged row's shared
# memory costs occupancy and the kernel stages only after the wait (same results bit for bit).
LC8_PREWAIT_MAX_ROWS = 8
# The MSA index scorer (ops/attention/m3_index_score) prefetches the older index-K pages into L2 before its grid wait
# at every batch it runs (more than FUSED_SCORE_SELECT_MAX_REQUESTS requests); it never runs its one-request
# early-issue schedule here.
INDEX_SCORE_EARLY_MAX_REQUESTS = 0

# The decode MXFP8 GEMVs, per projection (N, K): the rows each serves on an activation its producer quantized (the
# MoE boundary for the QKV, the attention merge for the o_proj).
_ALL_ROWS = frozenset(range(1, DECODE_MAX_TOKENS + 1))
MX_GEMV_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = {
    QKV_SPARSE: _ALL_ROWS,
    QKV: _ALL_ROWS,
    O_PROJ: _ALL_ROWS,
    DENSE_GATE_UP: _ALL_ROWS,
    DENSE_DOWN: _ALL_ROWS,
}
# On a bf16 activation the GEMV quantizes it in its prologue, which pays only up to 8 32-element blocks per thread:
# the most rows each projection takes.
MX_GEMV_BF16_MAX_TOKENS: Dict[Tuple[int, int], int] = {
    QKV_SPARSE: 16,
    QKV: 16,
    O_PROJ: 48,
    DENSE_GATE_UP: 16,
    DENSE_DOWN: 32,
}
# The Eagle3 drafter's BF16 linears (qkv, o_proj, gate_up, down, fc) through FlashInfer's fixed-tactic low-M kernels
# (ops/gemm/m3_draft_lowm): per (N, K), the rows its tactic table covers; more rows keep the in-tree GEMM.
DRAFT_LOWM_ROWS: Dict[Tuple[int, int], FrozenSet[int]] = {
    nk: frozenset(range(1, table[-1][0] + 1)) for nk, table in DRAFT_LOWM_TACTICS.items()
}

Key = Hashable


@dataclass(frozen=True)
class RowRange:
    """The shapes one fast path takes, and the catalog entry whose cells certify each of them.

    ``keys`` are the covered shapes: row counts for most paths, (query rows per request, requests) for the attention
    kernels, a variant name for thresholds. ``certifies`` maps a cell's spec of ``entry`` to the keys that cell
    drives (empty for a cell of another variant).
    """

    path: str
    keys: FrozenSet[Key]
    entry: OpWrapper
    certifies: Callable[[Mapping[str, Any]], Iterable[Key]]

    def covers(self, key: Key) -> bool:
        return key in self.keys

    def certified(self) -> FrozenSet[Key]:
        return frozenset(k for cell in self.entry.CELLS for k in self.certifies(cell.spec))


def _rows(lo: int, hi: int) -> FrozenSet[int]:
    return frozenset(range(lo, hi + 1))


def _spec_rows(field: str, **match: Any) -> Callable[[Mapping[str, Any]], Iterable[Key]]:
    """A cell's ``spec[field]`` when the cell's other fields match ``match``."""

    def certifies(spec: Mapping[str, Any]) -> Iterable[Key]:
        if any(spec.get(name) != value for name, value in match.items()):
            return ()
        return (spec[field],)

    return certifies


def _const(*keys: Key) -> Callable[[Mapping[str, Any]], Iterable[Key]]:
    return lambda spec: keys


def _sparse_decode_producer(spec: Mapping[str, Any]) -> Iterable[Key]:
    if not spec["identity"]:
        return ()
    rows = spec["decode_query_len"] * len(spec["seq_lens"])
    return ("multi_query" if rows >= SPARSE_DECODE_MQ_MIN_TOKENS else "per_row",)


def _long_select_widths(spec: Mapping[str, Any]) -> Iterable[Key]:
    """The key widths a cell's rows reach: every width up to its blocks' when rows of mixed lengths are drawn."""
    blocks = spec["num_blocks"]
    widest = 2 if blocks <= 2048 else 4 if blocks <= 4096 else 9
    if spec["valid"] == "mixed":
        return tuple(w for w in LONG_SELECT_KEY_WIDTHS if w <= widest)
    return (widest,)


EMBED_FIRSTNORM = RowRange(
    "embed_firstnorm",
    _rows(1, EMBED_FIRSTNORM_MAX_TOKENS),
    m3_embedding_firstnorm,
    _spec_rows("rows"),
)
SPARSE_DECODE_PRODUCER = RowRange(
    "sparse_decode_producer",
    frozenset({"per_row", "multi_query"}),
    m3_sparse_decode,
    _sparse_decode_producer,
)
# The long-context sparse decode: "prewait" up to LC8_PREWAIT_MAX_ROWS query rows, "plain" above.
SPARSE_DECODE_LC8 = RowRange(
    "m3_sparse_decode_lc8",
    frozenset({"prewait", "plain"}),
    m3_sparse_decode_lc8,
    lambda spec: ("prewait" if spec["prewait"] else "plain",),
)
# Every token count of a sparse layer on an NVFP4 KV cache at the TP8 heads (prefill and decode): the producer takes the
# geometry, not a row range.
PRODUCER = RowRange("m3_producer", frozenset({"nvfp4_tp8"}), m3_producer, _const("nvfp4_tp8"))
# Requests of a pure decode span whose blocks are scored and selected in one launch.
FUSED_SCORE_SELECT = RowRange(
    "m3_fused_score_select",
    _rows(1, FUSED_SCORE_SELECT_MAX_REQUESTS),
    m3_fused_score_select,
    _spec_rows("batch"),
)
INDEX_SCORE = RowRange(
    "m3_index_score",
    frozenset({"prefetch"}),
    m3_index_score,
    lambda spec: () if spec["early"] else ("prefetch",),
)
LONG_SELECT = RowRange(
    "long_select", frozenset(LONG_SELECT_KEY_WIDTHS), m3_select_long, _long_select_widths
)
MC_ALLREDUCE = RowRange(
    "mc_allreduce_norm",
    _rows(1, DECODE_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=0),
)
ATTN_PUSH = RowRange(
    "attn_push",
    _rows(1, ATTN_PUSH_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=AR_PUSH_MODE),
)
MOE_PUSH = RowRange(
    "moe_push",
    _rows(1, MOE_PUSH_MAX_TOKENS),
    m3_mc_allreduce_norm,
    _spec_rows("m", prepushed=AR_PUSH_MODE),
)
MC_FINALIZE = RowRange(
    "mc_finalize_allreduce_norm",
    _rows(MC_FINALIZE_MIN_TOKENS, DECODE_MAX_TOKENS),
    m3_mc_finalize_allreduce_norm,
    _spec_rows("m"),
)
MOE_ENGINE = RowRange(
    "m3_moe_engine", _rows(1, MOE_ENGINE_MAX_TOKENS), m3_moe_engine, _spec_rows("tokens")
)
ROUTER_GEMV = RowRange(
    "m3_router_gemv", _rows(1, ROUTER_GEMV_MAX_TOKENS), m3_router_gemv, _spec_rows("m")
)
ROUTER_GEMV_FP4 = RowRange(
    "m3_router_gemv_fp4", _rows(1, ROUTER_GEMV_MAX_TOKENS), m3_router_gemv_fp4, _spec_rows("m")
)
ROUTER_TC = RowRange(
    "m3_router_tc",
    _rows(ROUTER_GEMV_MAX_TOKENS + 1, ROUTER_TC_MAX_TOKENS),
    m3_router_tc,
    _spec_rows("m"),
)
# The GEMM paths are keyed (projection (N, K), rows): each projection runs its own split-K / ring / MMA
# configuration, so a row count is certified per projection.
MX_GEMV = RowRange(
    "m3_mx_gemv",
    frozenset((nk, m) for nk, rows in MX_GEMV_ROWS.items() for m in rows),
    m3_mx_gemv,
    lambda spec: ((spec["nk"], spec["m"]),),
)
MX_GEMV_BF16 = RowRange(
    "m3_mx_gemv_bf16",
    frozenset((nk, m) for nk, hi in MX_GEMV_BF16_MAX_TOKENS.items() for m in _rows(1, hi)),
    m3_mx_gemv_bf16,
    lambda spec: ((spec["nk"], spec["m"]),),
)
DRAFT_LOWM = RowRange(
    "m3_draft_lowm",
    frozenset((nk, m) for nk, rows in DRAFT_LOWM_ROWS.items() for m in rows),
    m3_draft_lowm,
    lambda spec: ((spec["nk"], spec["m"]),),
)
# (draft length, requests) of a greedy verify step.
SHARD_ACCEPT = RowRange(
    "m3_shard_accept",
    frozenset((SHARD_ACCEPT_DRAFT_LEN, n) for n in range(1, SHARD_ACCEPT_MAX_REQUESTS + 1)),
    m3_shard_accept,
    lambda spec: ((spec["draft_len"], spec["nreq"]),),
)

# Requests of a decode step whose dense-layer attention runs at a capped KV split.
DENSE_DECODE_SPLIT = RowRange(
    "m3_dense_decode",
    frozenset(DENSE_DECODE_MAX_SEQ_LEN_BY_BATCH),
    m3_dense_decode,
    lambda spec: (len(spec["kv_lens"]),) if spec["capped"] else (),
)

# (draft length, requests) of a fused draft loop, and the rows of its passes' draft input: the first pass's
# requests x (draft length + 1), one per request on the later passes.
DRAFT_STEP = RowRange(
    "m3_draft_step",
    frozenset((DRAFT_STEP_DRAFT_LEN, n) for n in range(1, DRAFT_STEP_MAX_REQUESTS + 1)),
    m3_draft_step,
    lambda spec: ((spec["draft_len"], spec["nreq"]),),
)
DRAFT_INPUT = RowRange(
    "m3_draft_input",
    frozenset(
        rows
        for n in range(1, DRAFT_STEP_MAX_REQUESTS + 1)
        for rows in (n, n * (DRAFT_STEP_DRAFT_LEN + 1))
    ),
    m3_draft_input,
    _spec_rows("m"),
)

FAST_PATHS: Tuple[RowRange, ...] = (
    EMBED_FIRSTNORM,
    DENSE_DECODE_SPLIT,
    SPARSE_DECODE_PRODUCER,
    SPARSE_DECODE_LC8,
    PRODUCER,
    FUSED_SCORE_SELECT,
    INDEX_SCORE,
    LONG_SELECT,
    MC_ALLREDUCE,
    ATTN_PUSH,
    MOE_PUSH,
    MC_FINALIZE,
    MOE_ENGINE,
    ROUTER_GEMV,
    ROUTER_GEMV_FP4,
    ROUTER_TC,
    MX_GEMV,
    MX_GEMV_BF16,
    DRAFT_LOWM,
    SHARD_ACCEPT,
    DRAFT_STEP,
    DRAFT_INPUT,
)

# --------------------------------------------------------------------------------------------------------------------
# Selection: pure functions of the step shape
# --------------------------------------------------------------------------------------------------------------------


def plan_for(num_tokens: int) -> DecodePlan:
    for plan in DECODE_PLANS:
        if num_tokens <= plan.max_tokens:
            return plan
    raise AssertionError(f"no decode plan covers {num_tokens} tokens")


def is_decode_step(num_tokens: int) -> bool:
    """Whether a pure generation step of ``num_tokens`` rows goes to the decode target."""
    return num_tokens <= DECODE_MAX_TOKENS


def moe_path(num_tokens: int) -> Optional[str]:
    """The MoE kernel of a decode step: the engine, the unfinalized MoE + finalize all-reduce, or None (generic)."""
    if MOE_ENGINE.covers(num_tokens):
        return MOE_ENGINE.path
    if MC_FINALIZE.covers(num_tokens):
        return MC_FINALIZE.path
    return None


def router_path(num_tokens: int) -> Optional[str]:
    """The router kernel when the MoE runs unfused: the fp32 GEMV, the tensor-core router, or None (built-in gate)."""
    if ROUTER_GEMV.covers(num_tokens):
        return ROUTER_GEMV.path
    if ROUTER_TC.covers(num_tokens):
        return ROUTER_TC.path
    return None


MX_GEMV_PATH = MX_GEMV.path
MX_GEMV_BF16_PATH = MX_GEMV_BF16.path
# FlashInfer's CuTe DSL MXFP8 GEMM on the producer's MXFP8 activation (rows the GEMV does not serve).
PREQUANT_GEMM_PATH = "flashinfer_mxfp8_gemm"
# The MXFP8 quantize (tuned backend) and FlashInfer's CuTe DSL GEMM: every row the decode kernels do not serve.
GENERIC_GEMM_PATH = "mxfp8_gemm"


def linear_path(n: int, k: int, num_tokens: int, prequant: bool) -> str:
    """The decode-graph MXFP8 GEMM of an [N, K] projection on ``num_tokens`` rows.

    ``prequant``: the producer published the activation in MXFP8.
    """
    if prequant:
        if num_tokens in MX_GEMV_ROWS.get((n, k), ()):
            return MX_GEMV_PATH
        return PREQUANT_GEMM_PATH
    if num_tokens <= MX_GEMV_BF16_MAX_TOKENS.get((n, k), 0):
        return MX_GEMV_BF16_PATH
    return GENERIC_GEMM_PATH


# --------------------------------------------------------------------------------------------------------------------
# Fallback policy
# --------------------------------------------------------------------------------------------------------------------


class FastPathUnavailable(RuntimeError):
    """A covered decode shape could not take its fast path under ``modeling_v2='require'``."""


_lock = threading.Lock()
_require = [False]
_logged: set = set()


def set_require(require: bool) -> None:
    """Whether a miss raises (``modeling_v2='require'``); set by the target at construction."""
    _require[0] = bool(require)


def require_mode() -> bool:
    return _require[0]


def _log_once(level: Callable[[str], None], path: str, reason: str, message: str) -> None:
    key = (path, reason)
    with _lock:
        if key in _logged:
            return
        _logged.add(key)
    level(message)


def missed(path: str, rows: Optional[int], reason: str) -> None:
    """``path`` covers this step but cannot run: raise under require, else log once; the caller falls back."""
    at = "" if rows is None else f" at {rows} rows"
    if _require[0]:
        raise FastPathUnavailable(
            f"modeling_v2='require': the MiniMax-M3 decode path {path!r} covers this step{at} but cannot run: "
            f"{reason}"
        )
    _log_once(
        logger.warning,
        path,
        reason,
        f"[M3 dispatch] {path}{at}: {reason}; falling back (logged once per reason)",
    )


def capture_missed(path: str, rows: Optional[int], what: str) -> None:
    """``what`` is built on eager steps but a CUDA-graph capture found it missing (see ``missed``)."""
    missed(path, rows, f"{what} is not built and a CUDA graph is being captured")


def uncovered(path: str, reason: str) -> None:
    """The step is outside ``path``'s covered shapes: log once per (path, reason); the generic path runs."""
    _log_once(logger.info, path, reason, f"[M3 dispatch] {path} not taken: {reason} (logged once)")


def reset_logged() -> None:
    """Forget which (path, reason) pairs were logged."""
    with _lock:
        _logged.clear()
