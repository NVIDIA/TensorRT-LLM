# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-model DSpark on this target (phase W0): the in-tree DSpark drafter and worker, unchanged but for the bitwise tail
items (TAIL_FORCEVEC, TAIL_DEFER_CAPTURE, TAIL_FORK, DRAFT_APPEND_FUSE,
TAILGLUE_B and MEMCPY86K below; TAIL_DSFOLD_FWD lives in modeling.py, DHEAD_NOSLICE and MARKOV16_B1 in
dspark_draft_head.py and dspark_markov_worker.py).

The target keeps its own forward; the shell hands the step to ``M3DSparkWorker``, which is the in-tree
``DSparkWorker`` plus the five calls the shell makes on any worker (``capture_hidden_states``,
``arm_capture_norm``, ``tail_eligible``, ``prebuild``, ``__call__``). In W0 none of them takes a fused path: the
captures are the in-tree DFlash captures, the target logits go through the logits processor and the in-tree
strict accept verifies them, so the speculation is the in-tree one by construction.

The drafter is the in-tree standalone DSpark drafter for the checkpoint's architecture, built through the draft
model registry. Its context K/V lives in the separate draft KV cache, which must keep the drafter's projection
dtype (BF16): a pool of another dtype is not bound and the worker falls back to a private arena sized by the
served sequence length. So the drafter never takes the target's KV quantization: neither the FP8 draft-KV
override (an Eagle3 need) nor the target's own KV algo (NVFP4 under ``kv_cache_config.dtype="nvfp4"``).
"""

from __future__ import annotations

import functools
import time
from typing import Optional, Set, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops import native_swiglu
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.comm import (
    allgather_cast as _allgather_cast,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    strict_accept as _strict_accept,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_block_tables import (
    enable_parallel_refresh,
    refresh_block_tables,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_advance import (
    advance_ctx_and_read_view,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_ctx_store import (
    fold_k_norm_weight,
    fused_ctx_store,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_drafter_inputs import (
    drafter_inputs as _drafter_inputs,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_forward_indices import (
    forward_indices,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_qk_norm_rope import (
    qk_norm_rope,
    qk_norm_rope_append,
)
from tensorrt_llm._torch.distributed import AllReduce, AllReduceStrategy, allgather
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import (
    external_drafter_config_kwargs,
    get_draft_model,
)
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.speculative.dspark import DSparkWorker
from tensorrt_llm._torch.speculative.interface import (
    _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
    _FORCE_ACCEPT_RNG_POOL_SIZE,
    _FORCE_ACCEPT_RNG_SLOT_STRIDE,
    should_use_separate_draft_kv_cache,
)
from tensorrt_llm.logger import logger

from .. import m31_capture_fork
from ..m31_tail_bookkeep import TailBookkeepMixin
from . import dspark_draft_head as _dhead8
from . import dspark_tailfold as _tfold
from .drafter_mxfp8 import route_mx_linears, route_projection_layout
from .dspark_markov_worker import MAX_FAST_BATCH, TAILBK_SPEC, S4WorkerMixin
from .dspark_t6 import route_lowm_linears, routes_line
from .dspark_tailfork import copy_captures, ctx_store_launch, ctx_store_prep, view_key
from .gen_glue import glue_disabled
from .kernels import m3_mc_allreduce, m3_nnt_fold, m3_report
from .m31_switches import m31_disabled, m31_enabled
from .ops.embedding.m3_embedding import m3_embedding
from .ops.gemm.m31_fc_shard import FcShard
from .ops.spec.m3_force_vec import m3_force_vec

# S9BVR: the drafter's row-parallel all-reduces (each layer's o_proj and MLP down projection) of up to this many
# rows run the one-shot all-reduce instead of AUTO. On this target AUTO resolves to NCCL at every row count: the AR
# tuner's profiling of the custom tactics fails here ("Total failed profiling tactics: 14"), so the
# decode-step drafter ARs ran as the NCCL ring. SM107 TP4 [rows, 6144] BF16, graph of 10 calls: 4 rows 16.46 ->
# 5.64 us per call, 32 rows 16.66 -> 6.46; error vs the FP32 sum 2.0e-3 against NCCL's 3.9e-3 (a different
# reduction order: not bitwise against NCCL). Wider calls (the drafter's context passes) keep AUTO. Never the two-shot
# all-reduce: 430 us at 4 rows here. TRTLLM_M31_DISABLE_S9BVR=1 restores AUTO at every row count.
# TP2: ON with the TP8 levers (MXDRAFT, S9b); the switch is parsed strictly (m31_switches).
S9BVR = not m31_disabled("S9BVR")
DRAFT_AR_ONESHOT_MAX_ROWS = 32 if S9BVR else 0  # capacity: fixed: the K3 c8 verify rows

# FRONT_CAPFORK (part of HOST_CAPTURE_TAIL): a decode capture layer's plain Eagle3 capture copy (no residual addend, not
# the final hidden state) runs on the family capture stream (``m31_capture_fork``), off the boundary -> QKV PDL edge;
# the target forward joins it after the decoder stack, before any capture consumer. Bitwise the same buffer. Rows
# above FRONT_CAPFORK_MAX_ROWS (wider than the K7 batch-1 verify) keep the inline copy. Default ON; kill switches
# TRTLLM_M31_DISABLE_FRONT_CAPFORK=1 or the group switch TRTLLM_M31_DISABLE_HOST_CAPTURE_TAIL=1, read at import.
FRONT_CAPFORK = not (m31_disabled("FRONT_CAPFORK") or m31_disabled("HOST_CAPTURE_TAIL"))
FRONT_CAPFORK_BASE_ROWS = 8  # capacity: fixed: the K7 batch-1 verify rows (1 x (7 + 1))
# FRONT_CAPFORK_WIDE: the K7 2..4-request verifies (16 / 24 / 32 rows) fork the same plain copy (the
# copy is row-generic: one cudaMemcpy2DAsync of the step's rows into the capture buffer's column block, bit for bit the
# in-line copy; the source is held until the join). Kill switch TRTLLM_M31_DISABLE_FRONT_CAPFORK_WIDE=1 (the
# batch-1 bound), read at import; the FRONT_CAPFORK switches above still turn the whole fork off.
FRONT_CAPFORK_WIDE = not (m31_disabled("FRONT_CAPFORK_WIDE") or m31_disabled("TP2WIDE_COMMGEMV"))
FRONT_CAPFORK_MAX_ROWS = (
    32 if FRONT_CAPFORK_WIDE else FRONT_CAPFORK_BASE_ROWS
)  # capacity: fixed: the K7 4-request verify (4 x (7 + 1) rows)


_CAPFORK_WIDE_LOGGED: list = []


def _note_capfork_wide(layer_id: int, hidden_states, residual) -> None:
    """FRONT_CAPFORK_WIDE's engagement: a forked capture past the batch-1 rows logs one line per process and counts
    its work (the same conditions ``m31_capture_fork.capture_or_inline`` forks on)."""
    if not (FRONT_CAPFORK and residual is None and layer_id != -1):  # -1: the final hidden state
        return
    rows = int(hidden_states.shape[0]) if hidden_states.dim() >= 1 else 0
    if not (FRONT_CAPFORK_BASE_ROWS < rows <= FRONT_CAPFORK_MAX_ROWS and hidden_states.is_cuda):
        return
    if not _CAPFORK_WIDE_LOGGED:
        _CAPFORK_WIDE_LOGGED.append(True)
        logger.info(
            f"[TP2WIDE] SMALL capfork engaged: the Eagle3 capture copy at {rows} rows forked off the boundary -> QKV "
            "PDL edge (kill TRTLLM_M31_DISABLE_FRONT_CAPFORK_WIDE=1)"
        )
    m3_report.work("FRONT_CAPFORK_WIDE", f"{rows} rows, layer {layer_id}")


class DraftOneshotAllReduce(nn.Module):
    """A row-parallel linear's all-reduce: the one-shot all-reduce up to ``max_rows`` rows, its original (AUTO)
    all-reduce above. The output buffer choice stays the original module's, so wider calls are unchanged."""

    def __init__(self, auto: AllReduce, dtype: torch.dtype, max_rows: int, name: str):
        super().__init__()
        self.auto = auto
        self.oneshot = AllReduce(
            mapping=auto.mapping, strategy=AllReduceStrategy.ONESHOT, dtype=dtype
        )
        self.max_rows = max_rows
        self.name = name
        self._logged: Set[int] = set()

    def uses_nccl_symmetric_memory_window(self) -> bool:
        return self.auto.uses_nccl_symmetric_memory_window()

    def forward(self, input: torch.Tensor, *args, **kwargs):
        rows = input.shape[0] if input.dim() > 1 else 1
        if rows > self.max_rows:
            return self.auto(input, *args, **kwargs)
        if rows not in self._logged:
            self._logged.add(rows)
            logger.info(
                f"[M3 dispatch] rank {self.auto.mapping.tp_rank}: S9b-VR-a drafter all-reduce ONESHOT taken at "
                f"{rows} rows ({self.name})"
            )
        m3_report.work(
            "S9BVR", f"{rows} rows, world {self.auto.mapping.tp_size}, ONESHOT ({self.name})"
        )
        return self.oneshot(input, *args, **kwargs)


def install_draft_oneshot_allreduce(draft: nn.Module, max_rows: int) -> int:
    """Wrap every row-parallel linear's AUTO all-reduce in the drafter (S9BVR); returns how many were wrapped."""
    if max_rows <= 0:
        return 0
    n = 0
    for name, mod in draft.named_modules():
        ar = getattr(mod, "all_reduce", None)
        if (
            isinstance(mod, Linear)
            and mod.tp_mode == TensorParallelMode.ROW
            and isinstance(ar, AllReduce)
            and ar.strategy == AllReduceStrategy.AUTO
            and ar.mapping.tp_size > 1
        ):
            mod.all_reduce = DraftOneshotAllReduce(ar, mod.dtype, max_rows, name)
            n += 1
    return n


# MXDRAFT (TP2: the TP8 default, ON): the drafter's qkv / o / gate_up / down GEMMs of the decode
# passes on this target's MXFP8 GEMV (drafter_mxfp8.py; drafter-only precision, gated on natural AL). TP4 measured it
# On by default. Kill switch
# TRTLLM_M31_DISABLE_DRAFTER_MXFP8=1, read at import.
MXDRAFT = not m31_disabled("DRAFTER_MXFP8")
# DRAFTER_HISLAYOUT: the drafter's qkv / o / gate_up / down on the family
# HISLAYOUT router (m31_drafter_layout) at the current TP2 GEMV tactics, projection-only (no fused SwiGLU + down), with
# the row-parallel o / down forward wrapped so the NCCL-window ROW branch no longer bypasses the GEMV. Kill switches
# TRTLLM_M31_DISABLE_DRAFTER_HISLAYOUT, read at import.
DRAFTER_HISLAYOUT = not m31_disabled("DRAFTER_HISLAYOUT")

# DRAFT_LOWM_GEMM (opt-in, TRTLLM_M31_ENABLE_DRAFT_LOWM_GEMM=1; dspark_t6): the drafter's decode-block GEMMs the MXFP8
# GEMV does not execute (the row-parallel o_proj and down, whose all-reduce takes the NCCL window output and so bypasses
# the MXFP8 hook) on FlashInfer's low-M BF16 kernels at tactics pinned per (N, K, M): o_proj and down at 7 and 14 rows
# (draft length 7, batches 1 and 2). Not bitwise against cuBLAS (drafter-only precision). With DRAFTER_HISLAYOUT on,
# those projections already run MXFP8 at these rows, so the low-M route finds no (projection, rows) left. Read at
# import.
T6 = m31_enabled("DRAFT_LOWM_GEMM")
# DRAFT_LOWM_GEMM takes every (projection, rows) the MXFP8 GEMV routed but does not execute
# (drafter_mxfp8.executed_rows: none for a row-parallel projection whose forward bypasses apply_linear). Kill switch
# TRTLLM_M31_DISABLE_DRAFT_LOWM_GEMM_HANDOFF=1: the low-M route leaves every projection the MXFP8 GEMV routed alone.
MXDRAFT_RP = not m31_disabled("DRAFT_LOWM_GEMM_HANDOFF")

# DRAFT_HEAD_MXFP8 (opt-in, TRTLLM_M31_ENABLE_DRAFT_HEAD_MXFP8=1; dspark_draft_head): the drafter head's Markov-chain
# shard GEMM ([B x K, 6144] x [100032, 6144]^T) on a separate MXFP8 copy of the shard padded to 100,096 rows (FlashInfer
# CuTe DSL quantize + CuTe DSL MXFP8 GEMM at a pinned tactic, logits sliced back to the shard) at the draft-length-7
# batch-1 / batch-2 rows; any other row count keeps the BF16 head. Not bitwise (drafter-only precision); adds about
# 634 MB per GPU at prebuild (before the KV cache is sized). Read at import.
DHEAD_MX8 = m31_enabled("DRAFT_HEAD_MXFP8")

# S9B-MC: S9b's fused drafter all-reduces + norms on the target's multicast all-reduce + norm (its 8-token bucket, the
# drafter-norm form: the unrounded fp32 sum normalized, opts bit 16) instead of the stock fused all-reduce; not
# bitwise, as S9b (a different summation order). The K7 c1 drafter block (8 rows) takes it; other row counts keep S9b's.
# TP2 (with MXDRAFT and S9BVR): ON by default. At world 2 the target's multicast op runs the
# drafter form on its 8-CTA cluster kernel (opts 23 / 31 = 16 | 7 / 16 | 15; the TP4 row kernel takes world 4 only):
# deterministic, the updated residual exact, the RMS statistic the cluster's sum (exact-class to FlashInfer's order,
# as the TP4 option-128 form). Rows past S9B_MC_MAX_ROWS keep S9b's fused all-reduce (S9BVR's one-shot below 33 rows).
# Kill switch TRTLLM_M31_DISABLE_DRAFT_AR_NORM_CLUSTER=1 (S9b's fused path at every row count). Read at import.
S9B_MC = not m31_disabled("DRAFT_AR_NORM_CLUSTER")
S9B_MC_BASE_ROWS = 8  # capacity: fixed: the drafter-norm form's 8-token bucket
# S9B-MC wide: the K7 2..4-request drafter blocks (14 / 21 / 28 rows, 7 per request) take the same
# drafter form on the 16- / 48-token buckets (option 31 on the cluster kernel: the build's
# m3_mc_allreduce.DRAFTER_WIDE_BUCKETS) instead of S9b's fused one-shot all-reduce + norm. Per row the same arithmetic
# as the shipped 8-row S9B-MC (the updated residual bitwise the stock BF16 all-reduce + add, the RMS statistic the
# cluster's sum): exact-class to the stock S9b path at these rows, not bitwise to it (as S9B-MC at 8 rows). An image
# whose kernel lacks the option-31 instances declines on its first wide call (every rank alike: same build) and keeps
# S9b there. Kill switch TRTLLM_M31_DISABLE_DRAFT_AR_NORM_CLUSTER_WIDE=1 (the 8-row bound), read at import.
S9B_MC_WIDE = S9B_MC and not (
    m31_disabled("DRAFT_AR_NORM_CLUSTER_WIDE") or m31_disabled("TP2WIDE_COMMGEMV")
)
S9B_MC_MAX_ROWS = (
    32 if S9B_MC_WIDE else S9B_MC_BASE_ROWS
)  # capacity: fixed: the K7 4-request drafter block (4 x 7 rows) fits 32
# Whether this process's build takes the wide drafter form: None until the first eager wide call decides it.
_S9B_MC_WIDE_STATE: dict = {"ok": None}


def mcag_unreachable(worker) -> Optional[str]:
    """Why no vocab-sharded draft logits reach the rejection sampler's all-gather that DRAFT_LOGITS_ALLGATHER_MC
    replaces (None: some can). Only the S4-logits path hands the stock sampler a rank's vocab shard of the draft logits;
    without it the drafter's head gathers its logits itself (gather_output), they arrive full-vocab, and the gather is
    never called (no DRAFT_LOGITS_ALLGATHER_MC work at TP2)."""
    if getattr(worker, "_s4", None) is None:
        return (
            "no vocab-sharded draft logits reach the rejection sampler: S4 is not set up, the drafter's gathered head "
            "returns full-vocab logits"
        )
    if not getattr(worker, "_s4_logits_on", False):
        return (
            "no vocab-sharded draft logits reach the rejection sampler: S4-logits is off, sampled steps take the "
            "in-tree chain on the gathered head's full-vocab logits"
        )
    return None


def _counted(item: str, fn, *extra):
    """``fn`` (a fused hook a DSpark worker installs) with a gate-e work counter: each call that did the work (returned
    neither None nor False) counts ``item`` and every ``(item, enabled)`` of ``extra`` whose flag is set. None stays
    None (the hook is not installed)."""
    if fn is None:
        return None

    @functools.wraps(fn)
    def run(*args, **kwargs):
        out = fn(*args, **kwargs)
        if out is not None and out is not False:
            m3_report.work(item, f"{getattr(fn, '__name__', item)} call")
            for name, on in extra:
                if on:
                    m3_report.work(name, f"inside {item}")
        return out

    return run


def _s9b_count_op(mapping, inner):
    """The fused AR + norm op installed whenever S9b is on: ``inner`` (S9B-MC's multicast op, or None) first; counts
    S9b's fused all-reduce + norm on every call (the stock fused path when ``inner`` declines) and S9B-MC's when it
    took the operands. Shapes decide, so every rank agrees."""

    def op(partial, residual, norm_weight, eps):
        out = inner(partial, residual, norm_weight, eps) if inner is not None else None
        rows = partial.shape[0] if partial.dim() == 2 else 0
        m3_report.work("S9B", f"{rows} rows, world {mapping.tp_size}")
        if out is not None:
            m3_report.work(
                "S9B_MC",
                f"{rows} rows, world {mapping.tp_size}, drafter form (opts 16) on the "
                + ("8-CTA cluster kernel" if mapping.tp_size != 4 else "row kernel"),
            )
        return out

    return op


def _s9b_mc_op(mapping):
    """The fused AR + norm op S9B-MC installs: (normed, residual + reduced) from the multicast all-reduce + norm, or
    None (the stock fused all-reduce runs) for operands it does not take. Shapes and the collectively created workspace
    decide, so every TP rank agrees."""

    def op(partial, residual, norm_weight, eps):
        if partial.dim() != 2 or not 1 <= partial.shape[0] <= S9B_MC_MAX_ROWS:
            return None
        rows = int(partial.shape[0])
        wide = rows > S9B_MC_BASE_ROWS
        if wide and not _s9b_mc_wide_ready():
            return None
        workspace = m3_mc_allreduce.get_workspace(mapping, partial.shape[-1])
        if workspace is None:
            return None
        try:
            out = m3_mc_allreduce.m3_mc_allreduce_norm(
                partial.contiguous(),
                residual.contiguous(),
                norm_weight,
                eps,
                workspace,
                use_pdl=get_env_enable_pdl(),
                drafter_norm=True,
                drafter_wide=wide,
            )
        except RuntimeError as exc:
            # The op's (bucket, option) check runs before any launch or exchange, on every rank alike.
            if not wide or "unsupported (bucket" not in str(exc):
                raise
            _S9B_MC_WIDE_STATE["ok"] = False
            logger.warning(
                f"[TP2WIDE] S9B_MC_WIDE off on rank {mapping.tp_rank}: this build's multicast op lacks the "
                f"16 / 48-token drafter form ({exc}); S9b's fused all-reduce runs past {S9B_MC_BASE_ROWS} rows"
            )
            return None
        if wide and out is not None:
            if _S9B_MC_WIDE_STATE["ok"] is None:
                logger.info(
                    f"[TP2WIDE] S9B_MC_WIDE engaged rank {mapping.tp_rank}: drafter all-reduce + norm at {rows} rows "
                    "on the multicast kernel's drafter form (kill TRTLLM_M31_DISABLE_DRAFT_AR_NORM_CLUSTER_WIDE=1)"
                )
            _S9B_MC_WIDE_STATE["ok"] = True
            m3_report.work("S9B_MC_WIDE", f"{rows} rows, world {mapping.tp_size}")
        return out

    return op


def _s9b_mc_wide_ready() -> bool:
    """Whether a wide S9B-MC call may run now: known good, or still unknown outside a CUDA-graph capture (the first
    eager call decides it). Host state set identically on every rank by the same calls in the same order."""
    ok = _S9B_MC_WIDE_STATE["ok"]
    if ok is None:
        return not torch.cuda.is_current_stream_capturing()
    return ok


# S10 (lean drafter glue) applies at every draft length on SM107 TP8: token-identical to the eager glue at K3
# and at K7 with the AllReduce autotune pinned (43 / 43 requests over c1 / c2 / c4, seeded T 1.0; an
# earlier K7 divergence ran with the autotune unpinned). TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE=1 turns it off.
S10_MAX_DRAFT_LEN = 63

# DRAFT_WINDOW_CAPPED_DECODE (opt-in, TRTLLM_M31_ENABLE_DRAFT_WINDOW_CAPPED_DECODE=1): the largest decode batch the
# window-capped max_seq_len applies to, by draft length; 0 = off (draft length 3: never; 7: batches 1 and 2).
S5_MAX_BATCH_BY_K = {3: 0, 7: 2}

# TAIL_FORCEVEC: a fractional forced
# acceptance (TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS, the benchmark knob) draws the step's forced counts with the
# one-launch m3_force_vec (the Eagle3 path's op) instead of the in-tree _apply_force_accepted_tokens' ~11 launches: the
# same counter bump, (counter, slot) hash, pool read, fp32 compare and clamp, written into the accepted counts in place
# (bitwise). It is the in-tree draw: one server-wide counter bumped once per call, not FORCE_AL_PER_REQUEST's
# per-request key (that switch drives only the Eagle3 fused verify). Default ON; kill switch
# TRTLLM_M31_DISABLE_TAIL_FORCEVEC=1. Read at import.
TAIL_FORCEVEC = not m31_disabled("TAIL_FORCEVEC")
# TAIL_DEFER_CAPTURE (opt-in, TRTLLM_M31_ENABLE_TAIL_DEFER_CAPTURE=1; under TAIL_FORK the copy runs in the fork): on a
# decode-only step, a capture layer whose MoE boundary is the MC finalize all-reduce (modeling's MC_FINALIZE branch:
# m3_mc_finalize_allreduce_norm, the 9..48-row decode steps, so the draft-length-7 batch-2 16-row step) is kept by
# reference instead of copied into the capture buffer, and the kept captures are written (the same bytes into the same
# columns: copy_captures, one concatenation when every capture layer was kept) just before the drafter first reads the
# buffer. No graph memcpy node then sits between a capture layer's boundary and the next layer's QKV GEMV. A capture is
# kept only when its tensor is, by identity, the finalize op's updated residual (modeling.is_mc_finalize_capture,
# published by the decode target around that one capture call): a fresh torch::empty_like output that every later
# launch of the step only reads (the next layer's boundary takes it as a const input). Every other capture (8-row steps,
# generic layers, mixed steps, the final hidden state) keeps the stock copy. FRONT_CAPFORK already takes these copies
# off the layer chain (capture stream), so this stays off by default; when on, a kept capture skips the capture-stream
# fork and TAIL_FORK's aux stream writes it. Read at import.
TAIL_DEFER_CAPTURE = m31_enabled("TAIL_DEFER_CAPTURE")


# TAIL_FORK (dspark_tailfork): at a
# decode-only step of n requests x (K + 1) verify rows, up to TAIL_FORK_MAX_ROWS (K7 batches 1 and 2: 8 and 16 rows),
# the drafter's acceptance-independent input work runs on an aux stream forked right after the verify head GEMM, beside
# the verify logits all-gather (A1) and the verify sampling, and joins where the stock path first reads the projected
# rows: the kept captures' copy into the capture buffer (TAIL_DEFER_CAPTURE's, one concatenation), the replicated fc +
# hidden norm, (TAIL_FORK_KV) the context K / V GEMM + k_norm of the DRAFTER_CTX_KV_STORE_FUSED
# store, and (TAIL_FORK_REFRESH) DRAFTER_BLOCK_TABLE_REFRESH_FUSED's block-table refresh. The TP4 target arms its fork
# only where FC_SHARD takes the fc (8 rows); this target arms it on the replicated fc at both of its row counts. SM107
# TP2 b2: chain B (capture concat .. k_norm) 57 us serial after the sampling, chain A (A1 + refresh + sampling + accept)
# 81 us mean. The same ops on the same operands, only on another stream and earlier, so bitwise as long as cuBLAS picks
# the same algorithm for the fc and the K / V GEMM on the aux stream as on the main stream (its reproducibility
# guarantee is per single active stream; a captured graph freezes the pick): the in-model output identity is the
# evidence, and a fork arm that alone loses identity calls for pinning those algorithms, not for a contention fix.
# Default ON; kill switches, read at import: TRTLLM_M31_DISABLE_TAIL_FORK=1 (all of it: the stock order),
# TRTLLM_M31_DISABLE_TAIL_FORK_KV=1 (the store's K / V GEMM + k_norm stay in the store),
# TRTLLM_M31_DISABLE_TAIL_FORK_REFRESH=1 (the refresh stays before the sampling), TRTLLM_M31_DISABLE_TAIL_FORK_B1=1 (no
# fork at the batch-1 step's 8 rows: a b1 loss can be shed without the b2 fork).
TAIL_FORK = not m31_disabled("TAIL_FORK")
TAIL_FORK_KV = TAIL_FORK and not m31_disabled("TAIL_FORK_KV")
TAIL_FORK_REFRESH = TAIL_FORK and not m31_disabled("TAIL_FORK_REFRESH")
TAIL_FORK_B1 = TAIL_FORK and not m31_disabled("TAIL_FORK_B1")
TAIL_FORK_MAX_ROWS = 16  # capacity: fixed: the K7 verify rows at batch 2 (b2t16 GEN); wider steps keep the stock order
TAIL_FORK_B1_ROWS = (
    8  # capacity: fixed: the K7 batch-1 verify rows (TAIL_FORK_B1 off: steps up to 8 rows unforked)
)
# TAIL_FORK_AFTER_VERIFY_GATHER (opt-in, TRTLLM_M31_ENABLE_TAIL_FORK_AFTER_VERIFY_GATHER=1): the fork point moves from
# right after the verify head GEMM to right after the verify logits' multicast all-gather (VERIFY_LOGITS_ALLGATHER_MC,
# "A1") launch, so the fork's work (its 1056-CTA fc first) waits for A1 to complete and cannot delay A1's CTAs; it then
# overlaps the verify sampling only. Bitwise either way. Read at import.
TAIL_FORK_AFTER_A1 = TAIL_FORK and m31_enabled("TAIL_FORK_AFTER_VERIFY_GATHER")
# TAIL_FORK_ISSUE_AFTER_VERIFY_GATHER (opt-in, TRTLLM_M31_ENABLE_TAIL_FORK_ISSUE_AFTER_VERIFY_GATHER=1; launch order
# only): the fork still depends only on the verify head GEMM (an event recorded right after it), but its kernels are
# issued after A1's launch, so A1's graph node and host launch come first while the fork keeps overlapping A1. Bitwise
# either way; TAIL_FORK_AFTER_VERIFY_GATHER wins over it. Read at import.
TAIL_FORK_A1_FIRST = (
    TAIL_FORK and not TAIL_FORK_AFTER_A1 and m31_enabled("TAIL_FORK_ISSUE_AFTER_VERIFY_GATHER")
)
# DRAFT_FC_SHARD (opt-in, TRTLLM_M31_ENABLE_DRAFT_FC_SHARD=1; ops/gemm/m31_fc_shard): the drafter's replicated fc as
# this rank's 3072-column shard (mv2_minimax_m3::minimax_m31_fc_shard_gemm at the full fc's pinned cuBLASLt algo) + an
# all-gather in rank order, bit for bit the full fc (the module's own forward) at its row counts (the draft-length
# verify rows of batches 1..MAX_FAST_BATCH up to TAIL_FORK_MAX_ROWS) by its boot self-check (probes at each of those row
# counts; a difference at one of them keeps the stock fc at that row count, on every rank). Inside the fork it halves
# each rank's fc weight read and CTAs (less contention with A1 and the verify sampling). Read at import.
FC_SHARD16 = m31_enabled("DRAFT_FC_SHARD")
# TAIL_FC_GATHER (with DRAFT_FC_SHARD, inside the fork): the shards' all-gather as one multicast launch (catalog
# allgather_f32 on the bf16 [rows, 3072] shard viewed as fp32 [rows, 1536]: the int32 words copied as is, in rank order
# along the last dim, so the bf16 view of the result is the NCCL all-gather's [rows, 6144] bit for bit) on its own
# workspace (not A1's), instead of the NCCL all-gather + concat. A fork that runs this multicast launch is forked after
# the verify logits' all-gather (A1, or its NCCL fallback), never beside it: two spin-waiting collectives on two streams
# have no proven progress order. The fork never all-gathers over NCCL: at row counts the multicast launch is not
# compiled for, the fork runs the replicated fc (bitwise the shard + gather by the self-check). Kill switch
# TRTLLM_M31_DISABLE_TAIL_FC_GATHER=1 (the replicated fc in the fork), read at import.
TAIL_FC_GATHER = TAIL_FORK and FC_SHARD16 and not m31_disabled("TAIL_FC_GATHER")


def _stock_projection(draft) -> bool:
    """Whether ``draft`` projects with the in-tree DFlash project_target_hidden (fc or its DRAFT_FC_SHARD shard, then
    hidden_norm), the one _tf_project reproduces with the multicast gather: the class's method is
    DFlashForCausalLM's own (inherited, not overridden)."""
    fn = getattr(type(draft), "project_target_hidden", None)
    return (
        getattr(fn, "__module__", None) == "tensorrt_llm._torch.models.modeling_dflash"
        and getattr(fn, "__qualname__", None) == "DFlashForCausalLM.project_target_hidden"
    )


class _TailStep:
    """One step's TAIL_FORK state (armed in arm_capture_norm, cleared when the worker's forward ends). The kept
    captures stay in the step's _DeferredCaptures (TAIL_DEFER_CAPTURE), which the fork copies."""

    __slots__ = (
        "spec",
        "draft",
        "attn",
        "rows",
        "total",
        "forked",
        "joined",
        "key",
        "proj",
        "prep",
        "used",
        "refreshed",
        "sharded",
        "stored",
        "skipped",
        "rebind",
        "point",
    )

    def __init__(self, spec, draft, attn, rows: int, total: int) -> None:
        self.spec = spec
        self.draft = draft
        self.attn = attn
        self.rows = rows  # the projected rows: generation requests x (K + 1)
        self.total = total  # the step's target tokens (get_hidden_states' bound)
        self.forked = False
        self.joined = False
        self.key = None  # view_key of the rows the fork projected
        self.proj = None  # the fork's project_target_hidden output
        self.prep = None  # the fork's ctx_store_prep
        self.used = False  # the drafter's projection returned the fork's rows
        # (tables, counts, num_seqs, attn_metadata) refreshed on the fork (TAIL_FORK_REFRESH). Holding both tensors the
        # fork's refresh writes keeps their blocks out of the caching allocator until the step-end join, even when a
        # draft KV cache rebind drops the worker's references while that write is still queued on the aux stream.
        self.refreshed = None
        self.sharded = None  # how the fork's fc ran: "replicated", "DRAFT_FC_SHARD shard + multicast gather", ...
        self.stored = None  # the store wrapper's use of the prep: None (not asked), True (its kernel only), False
        self.skipped = False  # the in-tree refresh was skipped for the fork's
        self.rebind = False  # the fork was joined before a draft KV cache rebind (its refresh then redone on main)
        self.point = None  # where the fork was launched (after the verify head GEMM, A1, ...)


class _DeferredCaptures:
    """One decode step's TAIL_DEFER_CAPTURE state (armed in arm_capture_norm, written before the drafter reads the
    capture buffer, dropped when the worker's forward ends)."""

    __slots__ = ("spec", "kept", "copied")

    def __init__(self, spec) -> None:
        self.spec = spec
        # [(capture index, boundary residual)] kept by reference: holding them keeps the caching allocator from handing
        # their memory to any later allocation of the step before the copy into the buffer is queued.
        self.kept = []
        # The kept captures were written into the buffer (no later capture of the step is kept).
        self.copied = False


def s10_applies(num_draft_tokens: int) -> bool:
    return num_draft_tokens <= S10_MAX_DRAFT_LEN


def install_draft_swiglu(draft_model) -> int:
    """Install the shared native activation on the drafter MLPs (GEN_NATIVE_GLUE). Kill:
    TRTLLM_M31_DISABLE_GEN_GLUE_NATIVE (or the item switch / TRTLLM_M31_DISABLE_DRAFT_SWIGLU)."""
    if glue_disabled("NATIVE_GLUE", "DRAFT_SWIGLU"):
        return 0
    layers = getattr(getattr(draft_model, "model", None), "layers", None) or []
    return native_swiglu.install(getattr(layer, "mlp", None) for layer in layers)


def s5_max_batch_for(num_draft_tokens: int) -> int:
    """K <= 3 takes the K3 entry, K >= 7 the K7 entry, and K in between the smaller of the two."""
    if num_draft_tokens <= 3:
        return S5_MAX_BATCH_BY_K[3]
    if num_draft_tokens >= 7:
        return S5_MAX_BATCH_BY_K[7]
    return min(S5_MAX_BATCH_BY_K.values())


def is_standalone_dspark(spec_config) -> bool:
    """Whether the run speculates with a standalone DSpark drafter (its own checkpoint, ``decoding_type: DSpark``)."""
    return bool(
        spec_config is not None
        and spec_config.spec_dec_mode.use_one_engine()
        and spec_config.spec_dec_mode.is_dspark()
        and not getattr(spec_config, "draft_is_embedded_in_target", False)
    )


class M3EmbedTokens(nn.Module):
    """The target's token embedding as the module the in-tree drafter borrows (``load_weights_from_target_model``).

    The drafter embeds the bonus and mask tokens through ``embed_tokens(ids)``; the target keeps its replicated
    table as a raw tensor and gathers it with ``m3_embedding``, so this module calls the same gather on the same
    table (an exact gather, so the drafter sees the rows the target embeds).
    """

    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self._core = [core]  # not a submodule: the table belongs to the target

    @property
    def weight(self) -> torch.Tensor:
        return self._core[0].w["embed"]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return m3_embedding(ids, self.weight)


def s10c_c_drafter_inputs(
    embed_tokens,
    accepted,
    num_accepted,
    slots,
    ctx_len,
    max_ctx,
    mask_token_id,
    trained_mask_embedding,
    block_size,
    tokens_per_request,
):
    """DRAFTER_INPUTS_FUSED: the drafter's step inputs in one launch (ops/dspark/dspark_drafter_inputs), for this
    target's embedding only: ``M3EmbedTokens`` is an exact gather of the replicated table, so the noise block is table
    rows (the
    trained mask row if the drafter keeps one). None for any other embedding: the stock path runs."""
    if not isinstance(embed_tokens, M3EmbedTokens):
        return None
    table = embed_tokens.weight
    mask_row = (
        trained_mask_embedding if trained_mask_embedding is not None else table[int(mask_token_id)]
    )
    return _drafter_inputs(
        accepted,
        num_accepted,
        slots,
        ctx_len,
        max_ctx,
        table,
        mask_row,
        block_size,
        tokens_per_request,
    )


# N3 A2 (switch TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED): the target's strict acceptance and next-step input tokens in
# one launch each (ops/dspark/strict_accept), bitwise to the in-tree torch path. Read once, at import.
N3A_DISABLED = m31_disabled("STRICT_ACCEPT_FUSED")
# N3A-KV (switch TRTLLM_M31_DISABLE_DRAFT_KV_REWIND_FUSED, and off with N3A): DFlash's draft-forward KV bookkeeping (the
# rewind amount 1 - num_accepted and kv_lens += 1) in one launch where the in-tree lines run, after the
# forced-acceptance override, bitwise (integer arithmetic). Read once, at import.
N3A_KV_DISABLED = N3A_DISABLED or m31_disabled("DRAFT_KV_REWIND_FUSED")
# N3 P-gid + P-cat (switch TRTLLM_M31_DISABLE_VERIFY_LOGITS_COPY_FUSED): the verify-row target logits without the
# identity gather_ids copy and with the vocab concat and the fp32 cast in one strided copy, bitwise to the logits
# processor. Read at import.
N3P_DISABLED = m31_disabled("VERIFY_LOGITS_COPY_FUSED")
# N3 A1 (switch TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC): the verify logits' vocab all-gather, concat and fp32
# cast in one launch over a multicast NVLink workspace (ops/comm/allgather_cast), bitwise to the NCCL all-gather +
# strided copy below. The verify rows (B x (K + 1), up to 64 rows: K3 batches 1..16, K7 batches 1..8). Read once, at
# import. Default on on SM107 since the SM107 op matrix + grid sweep certified it bitwise; the switch turns it off.
A1_DISABLED = m31_disabled("VERIFY_LOGITS_ALLGATHER_MC")
# A1_WIDE (agg TP2: K3 c46 / c56 / c64 = 184 / 224 / 256 verify rows): the A1 workspace holds 256 rows
# (allgather_cast.configure_capacity before the first workspace; 307 MB of multicast memory per rank at TP2). VR
# microbench (world 2, shard 100032): bitwise at 64 / 128 / 184 / 224 / 256 rows and 95..188 us per call faster than the
# NCCL all-gather + concat + cast past 64 rows (e.g. 256 rows 393.4 -> 204.7 us).
# TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC_WIDE=1 keeps 64 rows (the NCCL all-gather above them). Read once, at
# import.
A1_WIDE = not m31_disabled("VERIFY_LOGITS_ALLGATHER_MC_WIDE")
# The existing all-gather/cast also passed the 448-row capacity gate.
# Default ON, about 538 MB/rank at TP2. The new kill restores 256 rows; A1_WIDE still restores 64.
A1_WIDE_448 = A1_WIDE and not m31_disabled("VERIFY_LOGITS_ALLGATHER_MC_448")
A1_MAX_ROWS = 448 if A1_WIDE_448 else (256 if A1_WIDE else 64)
# Draft-head gathering reuses the prebuilt verify workspace; the kill keeps stock routing.
DRAFT_A1 = not m31_disabled("DRAFT_A1")
# A1 at draft lengths other than 3 (K7 included; switch TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN, K3
# keeps A1).
A1_K7_DISABLED = m31_disabled("VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN")
# DRAFT_LOGITS_ALLGATHER_MC (opt-in, TRTLLM_M31_ENABLE_DRAFT_LOGITS_ALLGATHER_MC=1): the rejection sampler's gather of
# the vocab-sharded fp32 draft logits (stock: the NCCL all-gather + a concat copy) as one launch on the
# VERIFY_LOGITS_ALLGATHER_MC multicast workspace (allgather_cast.allgather_f32: copied as is, bit for bit). Taken for
# the B * K draft rows up to the fp32 capacity while that workspace is set up; other shapes keep the stock all-gather.
# Off by default: with it on, two runs of the same requests sampled different tokens on a few long requests, which the
# stock all-gather does not. Read at import.
MCAG = m31_enabled("DRAFT_LOGITS_ALLGATHER_MC")
# DRAFT_APPEND_FUSE: on top of EX11, the same
# launch also writes the drafter block's K / V into the layer's paged cache (catalog qk_norm_rope_append: the same K
# bits, V copied, the same pages and slots as flashinfer's append_paged_kv_cache, HND), and the per-layer append launch
# is skipped (SM107 TP2 b2: AppendPagedKVCacheKernel 1.73-2.08 us x 5 layers per step). Taken only where the
# drafter appends through flashinfer's paged HND cache (append_supported); otherwise the per-layer append runs as
# before. Bitwise. Default ON; kill switch TRTLLM_M31_DISABLE_DRAFT_APPEND_FUSE=1, read at import.
DRAFT_APPEND_FUSE = not m31_disabled("DRAFT_APPEND_FUSE")
_APPEND_FUSE_LOGGED: Set[str] = set()


def _qk_norm_rope_append_logged(*args):
    """catalog qk_norm_rope_append, counting taken calls (EX11's counter too: its launch) and logging the first take
    under CUDA-graph capture once per process (the captured path), plus the first eager take."""
    out = qk_norm_rope_append(*args)
    if out is not None:
        rows = int(args[0].shape[0])
        m3_report.work("DRAFT_APPEND_FUSE", f"{rows} rows")
        m3_report.work("EX11", f"inside DRAFT_APPEND_FUSE, {rows} rows")
        kind = "captured" if torch.cuda.is_current_stream_capturing() else "eager"
        if kind not in _APPEND_FUSE_LOGGED:
            _APPEND_FUSE_LOGGED.add(kind)
            if kind == "captured":
                logger.info(
                    f"[M3 dispatch] DRAFT_APPEND_FUSE taken: drafter K / V append in the EX11 launch ({rows} rows, "
                    "under CUDA-graph capture)"
                )
            else:
                logger.info(
                    f"[M3 sm107 tp2] DRAFT_APPEND_FUSE at an eager step: drafter K / V append in the EX11 launch "
                    f"({rows} rows)"
                )
    return out


# TAILGLUE_B: on a decode-only step whose
# forward tail stores the step (TAIL_DSFOLD_FWD's M3DSparkStepFoldFwd), the next step's input tokens (N3 A2's
# next_new_tokens) and that tail's store + next-step staging (m3_fold_tail, the C++ foldTail this target shares with
# TP4) run as one CuTe DSL launch (kernels/m3_nnt_fold) right after the store wait the tail took: one launch and the gap
# between the two fewer per step (SM107 TP2 b2: next_new_tokens + foldTailKernel 2.72 us after the Markov
# chain). Integer copies of the same values into the same buffers: bitwise. Default ON; kill switch
# TRTLLM_M31_DISABLE_TAILGLUE_B=1 (the two launches), read at import. Off with N3A
# (TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED=1 restores the in-tree next_new_tokens on every step).
TAILGLUE_B = not m31_disabled("TAILGLUE_B") and not m31_disabled("STRICT_ACCEPT_FUSED")
_n3a_logged = False


def _n3a_log_once() -> None:
    global _n3a_logged
    if not _n3a_logged:
        _n3a_logged = True
        logger.info(
            "[M3 dispatch] n3_strict_accept taken: fused strict acceptance + next-step input tokens"
        )


_n3p_logged = False
_n3a_kv_logged = False


def _n3a_kv_log_once() -> None:
    global _n3a_kv_logged
    if not _n3a_kv_logged:
        _n3a_kv_logged = True
        logger.info(
            "[M3 dispatch] n3a_kv_rewind taken: draft-forward KV rewind + kv_lens bump in one launch"
        )


_a1_logged = set()


def _a1_log_once(key, message: str) -> None:
    if key not in _a1_logged:
        _a1_logged.add(key)
        logger.info(message)


def _n3p_log_once() -> None:
    global _n3p_logged
    if not _n3p_logged:
        _n3p_logged = True
        logger.info(
            "[M3 dispatch] n3p_target_logits taken: verify-row logits without the gather_ids copy and the concat"
        )


class _DraftA1Logits(nn.Module):
    """Gather the full-vocabulary draft head through a prebuilt A1 workspace."""

    def __init__(self, inner, head, workspace, rows: Tuple[int, ...]) -> None:
        super().__init__()
        self.inner = inner
        # The model owns its head; avoid registering it a second time here.
        self._head = [head]
        self._workspace = workspace
        self._rows = frozenset(rows)

    def forward(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
        if (
            lm_head is not self._head[0]
            or not return_context_logits
            or hidden_states.dim() != 2
            or hidden_states.dtype != torch.bfloat16
            or hidden_states.shape[0] not in self._rows
            or attn_metadata is None
            or attn_metadata.num_contexts != 0
        ):
            return self.inner(hidden_states, lm_head, attn_metadata, return_context_logits)
        shard = lm_head.apply_linear(hidden_states, None)
        m3_report.work("DRAFT_A1", f"{shard.shape[0]} draft rows, world {self._workspace['world']}")
        return _allgather_cast.allgather_cast(shard, **self._workspace)


# The DSpark worker's drafter hooks that prebuild arms (engine side, DFlashWorker.enable_*). On a TensorRT-LLM
# without them prebuild sets up nothing: every item it owns stays off and the in-tree DSpark worker paths run.
_WORKER_HOOKS = (
    "enable_lean_glue",
    "enable_window_capped_decode",
    "enable_full_slot_arena",
    "enable_fc_tp_split",
    "enable_fused_ar_norm",
    "enable_fused_ar_norm_op",
    "enable_fused_ctx_store",
    "enable_fused_drafter_inputs",
    "enable_fused_ctx_advance",
    "enable_fused_block_refresh",
    "enable_fused_forward_indices",
    "enable_fused_qk_norm_rope",
    "enable_fused_qk_norm_rope_append",
    "enable_draft_logits_gather_op",
    "enable_fused_kv_rewind",
)


class M3DSparkWorker(TailBookkeepMixin, S4WorkerMixin, DSparkWorker):
    """The in-tree ``DSparkWorker`` with the calls this target's shell makes on its worker. W1: the one-launch Markov
    chain (``S4WorkerMixin``, switch ``TRTLLM_M31_DISABLE_DRAFT_S4``). N3 A2: the strict acceptance and the next-step
    input tokens in one launch each (switch ``TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED``). N3 P / A1: the verify logits
    (switches ``TRTLLM_M31_DISABLE_VERIFY_LOGITS_COPY_FUSED`` / ``TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC``).
    TAILBK R1 / Z1 (``TailBookkeepMixin``, the TP4 worker's exact overrides; armed by modeling.TAILBK through
    ``spec_metadata.tail_bookkeep``). Everything else is the in-tree path."""

    def _static_arange(self, start: int, end: int) -> torch.Tensor:
        """The in-tree worker's cached arange; under S10 (lean glue) each call counts the S10 work counter (the lean
        glue's cached-index site, one of its strided-view sites)."""
        if getattr(self, "_lean_glue", False):
            m3_report.work("S10", f"lean glue arange [{start}, {end})")
        return super()._static_arange(start, end)

    def capture_hidden_states(
        self, spec_metadata, layer_id: int, hidden_states, residual=None
    ) -> None:
        """The in-tree DFlash capture: layer output (hidden + residual) into the static capture buffer; a decode
        capture layer's plain copy of up to FRONT_CAPFORK_MAX_ROWS rows runs on the capture stream (FRONT_CAPFORK).
        Under TAIL_DEFER_CAPTURE (opt-in), on a step armed for it, a capture whose tensor is the MC finalize
        boundary's updated residual is kept by reference instead, and written (the same bytes) before the drafter reads
        the buffer; such a capture does not reach the capture-stream fork."""

        st = getattr(
            self, "_dc", None
        )  # None off the opt-in (also on a worker built without this mixin's state)
        if (
            st is not None
            and residual is None
            and spec_metadata is st.spec
            and not st.copied
            and self._dc_keepable(spec_metadata, layer_id, hidden_states)
        ):
            st.kept.append((spec_metadata._layer_to_idx[layer_id], hidden_states))
            return

        m31_capture_fork.capture_or_inline(
            spec_metadata,
            layer_id,
            hidden_states,
            residual,
            "sm107_tp2",
            FRONT_CAPFORK,
            max_rows=FRONT_CAPFORK_MAX_ROWS,
        )
        if FRONT_CAPFORK_MAX_ROWS > FRONT_CAPFORK_BASE_ROWS:
            _note_capfork_wide(layer_id, hidden_states, residual)

    def arm_capture_norm(
        self, attn_metadata, spec_metadata, lm_head: nn.Module, draft_model
    ) -> None:
        """No capture norm: DSpark's fc reads the raw captured layers. Arms (or clears) this step's
        TAIL_DEFER_CAPTURE and TAIL_FORK."""
        self._dc = self._dc_arm(attn_metadata, spec_metadata) if self._dc_on else None
        self._tf = (
            self._tf_arm(attn_metadata, spec_metadata, draft_model)
            if self._tf_aux is not None
            else None
        )
        return None

    # ------------------------------------------------------------------------------------- TAIL_DEFER_CAPTURE

    _dc = None  # this step's _DeferredCaptures, or None
    _dc_on = False  # TAIL_DEFER_CAPTURE set up (prebuild)
    _dc_logged = ()  # the (rows, kept, one concatenation) shapes whose engagement line this rank logged

    def _dc_arm(self, attn_metadata, spec_metadata):
        """This step's deferred-capture state when the step is one TAIL_DEFER_CAPTURE serves, else None: a
        decode-only, non-ragged drafting step with a DFlash capture buffer. Host state only (the same on every rank,
        fixed per captured graph); which captures are kept is decided per capture (_dc_keepable)."""
        if spec_metadata is None or attn_metadata is None:
            return None
        if getattr(spec_metadata, "is_ragged_verify", False):
            return None
        if attn_metadata.num_contexts != 0:
            return None
        if int(getattr(spec_metadata, "runtime_draft_len", 0) or 0) <= 0:
            return None
        if getattr(spec_metadata, "captured_hidden_states", None) is None or not getattr(
            spec_metadata, "layers_to_capture", None
        ):
            return None
        return _DeferredCaptures(spec_metadata)

    def _dc_keepable(self, spec_metadata, layer_id: int, hidden_states) -> bool:
        """Whether this capture can be kept by reference: the decode capture layer's MC finalize boundary residual
        (modeling.is_mc_finalize_capture: this exact tensor, during its capture call), a capture layer, the buffer's
        dtype and width, every row fits."""
        from . import modeling as _modeling  # imported here: modeling imports this module

        if not isinstance(hidden_states, torch.Tensor) or not _modeling.is_mc_finalize_capture(
            hidden_states
        ):
            return False
        idx = getattr(spec_metadata, "_layer_to_idx", {}).get(layer_id)
        buf = spec_metadata.captured_hidden_states
        return (
            idx is not None
            and hidden_states.dim() == 2
            and hidden_states.is_contiguous()
            and hidden_states.dtype == buf.dtype
            and hidden_states.device == buf.device
            and hidden_states.shape[1] == spec_metadata.hidden_size
            and hidden_states.shape[0] <= buf.shape[0]
        )

    _dc_fork_logged = ()  # the (rows, kept, one concatenation, captured) in-fork takes whose line this rank logged

    def _dc_flush(
        self, st: _DeferredCaptures, forked: bool = False, captured: bool = False
    ) -> None:
        """Write the kept captures into the buffer on the current stream (copy_captures), once; later captures of the
        step (none on the stock path) are copied as they come. ``forked``: the call runs on TAIL_FORK's stream (the
        same copy, earlier; its own engagement line, which says whether the step is being captured: ``captured``)."""
        if st.kept and not st.copied:
            spec = st.spec
            rows = st.kept[0][1].shape[0]
            cat = copy_captures(
                spec.captured_hidden_states,
                rows,
                spec.hidden_size,
                spec.num_capture_layers,
                st.kept,
            )
            key = (rows, len(st.kept), cat)
            rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
            if forked:
                if (*key, captured) not in self._dc_fork_logged:
                    self._dc_fork_logged = (*self._dc_fork_logged, (*key, captured))
                    logger.info(
                        f"[M3 sm107 tp2] fast path engaged: TAIL_DEFER_CAPTURE taken in the TAIL_FORK fork ({rows} "
                        f"rows, {len(st.kept)} of {spec.num_capture_layers} captures, "
                        f"{'one concatenation' if cat else 'slice copies'}, "
                        f"{'under CUDA-graph capture' if captured else 'eager'}): MC finalize boundary residuals kept "
                        f"by reference and written on the fork's stream before the drafter's fc reads them on rank "
                        f"{rank}"
                    )
            elif key not in self._dc_logged:
                self._dc_logged = (*self._dc_logged, key)
                logger.info(
                    f"[M3 sm107 tp2] fast path engaged: TAIL_DEFER_CAPTURE taken ({rows} rows, {len(st.kept)} of "
                    f"{spec.num_capture_layers} captures, "
                    f"{'one concatenation' if cat else 'slice copies'}): MC finalize boundary residuals kept by "
                    f"reference and written before the drafter reads them on rank "
                    f"{rank}"
                )
            m3_report.work(
                "TAIL_DEFER_CAPTURE",
                f"{len(st.kept)} captures, {rows} rows, {'one concatenation' if cat else 'slice copies'}"
                + (", in the fork" if forked else ""),
            )
        st.copied = True

    # ------------------------------------------------------------------------------------------------ TAIL_FORK

    _tf = None  # this step's _TailStep, or None
    _tf_aux = None  # the fork's stream (prebuild, TAIL_FORK)
    _tf_gather = (
        None  # TAIL_FC_GATHER: (allgather_f32 workspace args, rows, fp32 shard columns) (prebuild)
    )
    _tf_logged = (
        False  # the TAIL_FORK engagement line (first fork under CUDA-graph capture) was logged
    )
    _tf_rows_logged = ()  # the (rows, captured) forks whose per-row-count line this rank logged
    _tf_kv = False  # TAIL_FORK_KV with the fused ctx store's wrapper installed (prebuild): the fork preps its K / V
    _tf_consumed_logged = ()  # the row counts whose captured "TAIL_FORK consumed" key line this rank logged
    _tf_fallback_logged = ()  # the fork parts whose "TAIL_FORK fallback" line this rank logged
    _tf_rebind_logged = False  # the "joined before the draft KV cache rebind" line was logged

    def _tf_arm(self, attn_metadata, spec_metadata, draft_model):
        """This step's fork state when the step is one TAIL_FORK serves, else None. Host state and shapes only (the
        same on every rank, fixed per captured graph): a decode-only, non-ragged drafting step of n requests x (K + 1)
        rows (at most TAIL_FORK_MAX_ROWS), a DFlash capture buffer, a drafter with the stock fc (no DRAFT_FC_TP_SPLIT
        split; the replicated fc, or DRAFT_FC_SHARD's shard when it takes these rows), the context bound to the draft
        pool."""
        if draft_model is None or spec_metadata is None or attn_metadata is None:
            return None
        if getattr(spec_metadata, "is_ragged_verify", False):
            return None
        if attn_metadata.num_contexts != 0:
            return None
        num_seqs = int(attn_metadata.num_seqs)
        if num_seqs <= 0 or int(getattr(spec_metadata, "runtime_draft_len", 0) or 0) <= 0:
            return None
        rows = num_seqs * int(getattr(self, "_draft_tokens_per_req", 0) or 0)
        total = int(attn_metadata.num_tokens)
        buf = getattr(spec_metadata, "captured_hidden_states", None)
        if buf is None or not getattr(spec_metadata, "layers_to_capture", None) or rows != total:
            return None
        if rows <= 0 or rows > TAIL_FORK_MAX_ROWS or total > buf.shape[0]:
            return None
        if rows <= TAIL_FORK_B1_ROWS and not TAIL_FORK_B1:
            return None
        if getattr(draft_model, "lilicorr", None) is not None:
            return None
        if getattr(draft_model, "_dflash_fc_split", None) is not None:
            return None
        if (
            getattr(draft_model, "fc", None) is None
            or getattr(draft_model, "hidden_norm", None) is None
        ):
            return None
        shard = getattr(draft_model, "_dflash_fc_shard", None)
        if (
            shard is not None
            and shard.applies(draft_model.fc, rows)
            and not _stock_projection(draft_model)
        ):
            return (
                None  # a projection of its own might all-gather the shard over NCCL inside the fork
            )
        if not (
            getattr(self, "_ctx_buf_inited", False)
            and getattr(self, "_ctx_paged", False)
            and getattr(self, "_ctx_block_tables", None) is not None
        ):
            return None
        return _TailStep(spec_metadata, draft_model, attn_metadata, rows, total)

    def _tf_launch(self, after=None, point: str = "head") -> None:
        """Fork this step's drafter input work onto the aux stream: after everything queued on the current stream
        (the target forward, its captures, the verify head GEMM and, at a later ``point``, the verify logits'
        all-gather), or, with ``after``, after that event only (TAIL_FORK_ISSUE_AFTER_VERIFY_GATHER: the head
        GEMM's)."""
        st = self._tf
        if st is None or st.forked:
            return
        dc = self._dc
        if dc is not None and dc.spec is not st.spec:
            return  # never: both are armed from the same step's metadata
        spec, draft = st.spec, st.draft
        aux = self._tf_aux
        captured = torch.cuda.is_current_stream_capturing()
        if after is None:
            aux.wait_stream(torch.cuda.current_stream())
        else:
            aux.wait_event(after)
        with torch.cuda.stream(aux):
            # TAIL_DEFER_CAPTURE's kept captures, written before the fc reads the buffer (the same bytes).
            if dc is not None:
                self._dc_flush(dc, forked=True, captured=captured)
            hs = spec.get_hidden_states(st.total)
            gen = hs[0 : st.rows]
            gen = gen.reshape(-1, gen.shape[-1])
            proj, st.sharded = self._tf_project(draft, gen)
            prep = None
            kv_buf = getattr(self, "_ctx_kv_buf", None)
            if self._tf_kv and kv_buf is not None:
                first = kv_buf if isinstance(kv_buf, torch.Tensor) else kv_buf[0]
                if proj.dtype == first.dtype:
                    prep = ctx_store_prep(draft, proj)
            attn = st.attn
            tables = getattr(self, "_ctx_block_tables", None)
            counts = getattr(self, "_ctx_block_counts", None)
            if TAIL_FORK_REFRESH and attn is not None and tables is not None:
                num_seqs = int(attn.num_seqs)
                if super()._refresh_ctx_block_tables(attn, num_seqs):
                    # Both written tensors held until the step-end join (see _TailStep.refreshed).
                    st.refreshed = (tables, counts, num_seqs, attn)
        st.forked = True
        st.point = point
        st.key = view_key(gen)
        st.proj = proj
        st.prep = prep
        self._tf_log(st)

    def _tf_log(self, st: _TailStep) -> None:
        """The engagement lines of a fork: TAIL_FORK's key line once per rank, at the first fork under CUDA-graph
        capture (the captured path; an eager fork never logs it), and one line per (rows, captured) per rank; the work
        counter on every fork."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        captured = torch.cuda.is_current_stream_capturing()
        what = (
            f"drafter fc ({st.sharded}){' + ctx K/V prep' if st.prep is not None else ''}"
            f"{' + block-table refresh' if st.refreshed is not None else ''}"
        )
        if captured and not self._tf_logged:
            self._tf_logged = True
            logger.info(
                f"[M3 dispatch] TAIL_FORK taken: {what} on the aux stream after the verify {st.point} ({st.rows} rows, "
                f"under CUDA-graph capture) on rank {rank}"
            )
        key = (st.rows, captured)
        if key not in self._tf_rows_logged:
            self._tf_rows_logged = (*self._tf_rows_logged, key)
            logger.info(
                f"[M3 sm107 tp2] TAIL_FORK at {st.rows} rows ({'captured' if captured else 'eager'}): {what}, after "
                f"the verify {st.point} on rank {rank}"
            )
        m3_report.work("TAIL_FORK", f"{st.rows} rows, {what}")

    def _tf_gather_takes(self, rows: int) -> bool:
        g = self._tf_gather
        return g is not None and rows in g[1]

    def _tf_mc(self, st: _TailStep) -> bool:
        """Whether this step's fork runs TAIL_FC_GATHER's multicast all-gather (DRAFT_FC_SHARD's shard takes its rows
        and the multicast launch is compiled for them): such a fork is launched after the verify logits' all-gather."""
        draft = st.draft
        shard = getattr(draft, "_dflash_fc_shard", None)
        return (
            shard is not None
            and shard.applies(draft.fc, st.rows)
            and self._tf_gather_takes(st.rows)
        )

    def _tf_late(self, a1: bool) -> bool:
        """Whether this step's fork is launched after the verify logits' all-gather instead of right after the head
        GEMM: TAIL_FORK_AFTER_VERIFY_GATHER (with A1 taken), or a fork with TAIL_FC_GATHER's multicast launch (never
        beside A1's or the NCCL all-gather)."""
        st = self._tf
        if st is None or st.forked:
            return False
        return (TAIL_FORK_AFTER_A1 and a1) or self._tf_mc(st)

    def _tf_project(self, draft, hidden_states):
        """(the drafter's projection of ``hidden_states``, how the fc ran), with no collective but TAIL_FC_GATHER's
        multicast launch: where DRAFT_FC_SHARD's shard takes these rows, its shard GEMM + that multicast all-gather when
        it is compiled for them (the same partial GEMM, the same gathered bytes, the same hidden norm), else the
        replicated fc (bitwise the shard + gather at these rows by DRAFT_FC_SHARD's self-check), never the NCCL
        all-gather (_tf_arm armed such a step only with the stock projection); without the shard, the drafter's own
        projection (the replicated fc)."""
        shard = getattr(draft, "_dflash_fc_shard", None)
        rows = hidden_states.shape[0]
        if shard is None or not shard.applies(draft.fc, rows):
            return draft.project_target_hidden(hidden_states), "replicated"
        x = hidden_states.to(draft.fc.weight.dtype)  # project_target_hidden's first line
        if self._tf_gather_takes(rows):
            ws, _, cols = self._tf_gather
            part = shard.partial(x)
            if (
                part.dtype == torch.bfloat16
                and part.is_contiguous()
                and part.shape == (rows, 2 * cols)
                and part.data_ptr() % 16 == 0
                and _allgather_cast.is_compiled(
                    rows, cols, ws["world"], device=part.device, fp32=True
                )
            ):
                full = _allgather_cast.allgather_f32(part.view(torch.float32), **ws)
                return (
                    draft.hidden_norm(full.view(torch.bfloat16)),
                    "DRAFT_FC_SHARD shard + multicast gather",
                )
        return (
            draft.hidden_norm(draft.fc(x)),
            "replicated: DRAFT_FC_SHARD's multicast gather not compiled at these rows",
        )

    def _refresh_ctx_block_tables(self, attn_metadata, num_seqs: int) -> bool:
        """The in-tree refresh, unless this step's fork already refreshed the same tables for the same step."""
        st = self._tf
        if st is not None and st.refreshed is not None:
            tables, counts, n, attn = st.refreshed
            if (
                attn is attn_metadata
                and n == int(num_seqs)
                and tables is getattr(self, "_ctx_block_tables", None)
                and counts is getattr(self, "_ctx_block_counts", None)
            ):
                st.skipped = True
                return True
        return super()._refresh_ctx_block_tables(attn_metadata, num_seqs)

    def _lazy_init_ctx_buffers(
        self, draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager=None
    ):
        """The in-tree lazy init. A forked step is joined first when the init is about to rebind (the in-tree
        condition: buffers set up for another draft KV cache manager than this step's, as when KV cache estimation's
        probe pass hands over to the final executor): the rebind drops the block tables and counts the fork's refresh
        writes and allocates its new buffers on this stream, so the fork's queued writes land first. The step then
        refreshes the new tables on the main stream (the override above no longer matches them)."""
        st = self._tf
        if (
            st is not None
            and st.forked
            and not st.joined
            and getattr(self, "_ctx_buf_inited", False)
        ):
            mgr = draft_kv_cache_manager
            if mgr is not None and not hasattr(attn_metadata, "draft_kv_cache_block_offsets"):
                mgr = None  # the in-tree init's private-arena case
            if getattr(self, "_ctx_kv_manager", None) is not mgr:
                self._tf_join(st)
                st.rebind = True
                m3_report.work("TAIL_FORK_REBIND", f"{st.rows} rows")
                if not self._tf_rebind_logged:
                    self._tf_rebind_logged = True
                    logger.info(
                        f"[M3 sm107 tp2] TAIL_FORK joined before the draft KV cache rebind ({st.rows} rows, "
                        f"{'captured' if torch.cuda.is_current_stream_capturing() else 'eager'}): the new block "
                        f"tables are refreshed on the main stream on rank "
                        f"{getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
                    )
        return super()._lazy_init_ctx_buffers(
            draft_model, spec_metadata, attn_metadata, draft_kv_cache_manager
        )

    def _tf_account(self, st: _TailStep) -> None:
        """A forked step's end (after the join, when the forward returned): which forked parts the step used. All of
        them (the drafter's projection returned the fork's rows; the store wrapper ran only its kernel on the prep; the
        in-tree refresh was skipped for the fork's): the TAIL_FORK_CONSUMED counter and, once per rank and row count at
        the first such step under CUDA-graph capture, the key line "TAIL_FORK consumed". A part launched but redone on
        the main stream: the TAIL_FORK_FALLBACK counter and one "TAIL_FORK fallback" line per rank and part (the
        refresh of a rebind step excepted: redone by design)."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        captured = torch.cuda.is_current_stream_capturing()
        parts = [
            ("projection", st.used, "projection unused: the drafter projected on the main stream")
        ]
        if st.prep is not None:
            parts.append(("store prep", st.stored is True,
                          "TAIL_FORK_KV prep unused: the store ran its K / V GEMM on the main stream"))  # fmt: skip
        if st.refreshed is not None and not st.rebind:
            parts.append(("refresh", st.skipped, "TAIL_FORK_REFRESH redone on main"))
        what = " + ".join(name for name, _, _ in parts)
        missed = [(name, why) for name, ok, why in parts if not ok]
        if not missed:
            m3_report.work("TAIL_FORK_CONSUMED", f"{st.rows} rows, {what}")
            if captured and st.rows not in self._tf_consumed_logged:
                self._tf_consumed_logged = (*self._tf_consumed_logged, st.rows)
                logger.info(
                    f"[M3 dispatch] TAIL_FORK consumed: {what} ({st.rows} rows, under CUDA-graph capture) on rank "
                    f"{rank}"
                )
            return
        for name, why in missed:
            m3_report.work("TAIL_FORK_FALLBACK", f"{name}, {st.rows} rows")
            if name not in self._tf_fallback_logged:
                self._tf_fallback_logged = (*self._tf_fallback_logged, name)
                logger.info(
                    f"[M3 sm107 tp2] TAIL_FORK fallback: {why} ({st.rows} rows, "
                    f"{'captured' if captured else 'eager'}) on rank {rank}"
                )

    def _tf_join(self, st: _TailStep) -> None:
        """The current stream waits for the fork (once). Outside capture, the fork's outputs read on the current stream
        are recorded on it, so their blocks are not handed to later aux work while those reads are queued."""
        if not st.forked or st.joined:
            return
        main = torch.cuda.current_stream()
        main.wait_stream(self._tf_aux)
        st.joined = True
        if not torch.cuda.is_current_stream_capturing():
            for t in (
                st.proj,
                *(() if st.prep is None else (st.prep.kv_flat, st.prep.k)),
            ):
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    t.record_stream(main)

    def prepare_1st_drafter_inputs(self, *args, **kwargs):
        """``_tf_prepare_1st_drafter_inputs``; on a decode-only step MEMCPY86K then names the step's
        DRAFTER_INPUTS_FUSED noise block
        to the drafter (its next forward's residual stream starts in it)."""
        inputs = self._tf_prepare_1st_drafter_inputs(*args, **kwargs)
        if self._tf86_on and self._decode_step:
            draft = kwargs.get("draft_model", args[7] if len(args) > 7 else None)
            self._tf86_named = _tfold.own_noise(draft, inputs, self._tfold_inputs)
            self._tf86_step = _tfold.capturing()
            self._tf86_capt = self._tf86_named and self._tf86_step
        return inputs

    def _tf_prepare_1st_drafter_inputs(self, *args, **kwargs):
        """The in-tree first drafter inputs (the drafter's fc reads the capture buffer here first), after this step's
        kept captures are written into it (by the fork, or here); on a forked step the drafter's project_target_hidden
        returns the fork's rows (after the join) for exactly the rows the fork projected."""
        dc = self._dc
        if dc is not None:
            self._dc_flush(dc)  # nothing left to write when the fork copied them
        st = self._tf
        if st is None or not st.forked:
            return super().prepare_1st_drafter_inputs(*args, **kwargs)
        draft = kwargs.get("draft_model", args[7] if len(args) > 7 else None)
        if draft is not st.draft:
            self._tf_join(st)
            return super().prepare_1st_drafter_inputs(*args, **kwargs)
        stock = type(draft).project_target_hidden

        def project(hidden_states, _st=st, _draft=draft, _stock=stock):
            self._tf_join(_st)
            if not _st.used and view_key(hidden_states) == _st.key:
                _st.used = True
                return _st.proj
            return _stock(_draft, hidden_states)

        draft.project_target_hidden = project
        try:
            return super().prepare_1st_drafter_inputs(*args, **kwargs)
        finally:
            del draft.project_target_hidden
            self._tf_join(st)

    def _tf_ctx_store(
        self,
        draft_model,
        projected,
        positions,
        num_accepted,
        request_rows,
        cols,
        block_table,
        kv_buf,
        tokens_per_request,
        *,
        capacity_counts=None,
        max_col=None,
    ):
        """DRAFTER_CTX_KV_STORE_FUSED's fused store; on a forked step whose fork prepared these projected rows, only its
        store kernel."""
        st = self._tf
        if st is not None and st.prep is not None and st.joined:
            done = ctx_store_launch(
                st.prep,
                draft_model,
                projected,
                positions,
                num_accepted,
                request_rows,
                cols,
                block_table,
                kv_buf,
                tokens_per_request,
                capacity_counts=capacity_counts,
                max_col=max_col,
            )
            if done is not None:
                st.stored = True
                return done
            if st.stored is None:
                st.stored = False
        return fused_ctx_store(
            draft_model,
            projected,
            positions,
            num_accepted,
            request_rows,
            cols,
            block_table,
            kv_buf,
            tokens_per_request,
            capacity_counts=capacity_counts,
            max_col=max_col,
        )

    def _tf_prebuild(self, draft_model) -> None:
        """TAIL_FORK's stream (rank-local) and TAIL_FC_GATHER's workspace + compiles (collective, after A1's workspace
        fixed the multicast capacity), outside capture, on every rank; then its set-up line."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        if TAIL_FORK and draft_model is not None and torch.cuda.is_available():
            self._tf_aux = torch.cuda.Stream()
        shard = getattr(draft_model, "_dflash_fc_shard", None) if draft_model is not None else None
        if self._tf_aux is not None and TAIL_FC_GATHER and shard is not None:
            self._tf_gather = self._tf_prebuild_gather(draft_model)
        on = self._tf_aux is not None
        point = (
            "A1 all-gather"
            if TAIL_FORK_AFTER_A1
            else ("head, issued after A1's launch" if TAIL_FORK_A1_FIRST else "head")
        )
        logger.info(
            f"M3.1 DSpark TAIL_FORK: {'on' if on else 'off'} (decode steps up to {TAIL_FORK_MAX_ROWS} rows, the "
            f"{TAIL_FORK_B1_ROWS}-row batch-1 step {'on' if TAIL_FORK_B1 else 'off'}; fork after the verify {point}; "
            f"KV prep {'on' if self._tf_kv and on else 'off'}, refresh {'on' if TAIL_FORK_REFRESH and on else 'off'}, "
            f"captures in the fork {'on' if self._dc_on and on else 'off'}, fc "
            f"{'DRAFT_FC_SHARD' if shard is not None else 'replicated'}, fc gather in the fork "
            + (
                f"multicast at rows {self._tf_gather[1]}, forked after the verify logits' all-gather"
                if self._tf_gather is not None
                else ("none: the replicated fc" if shard is not None else "none")
            )
            + f") on rank {rank}"
            + ("" if TAIL_FORK else " (TRTLLM_M31_DISABLE_TAIL_FORK set)")
        )

    def _tf_prebuild_gather(self, draft_model):
        """TAIL_FC_GATHER's workspace and compile (collective: every rank of the TP group runs it at the same point of
        prebuild, and every rank keeps it only if every rank compiled): (workspace args, rows, fp32 columns) or None."""
        mapping = getattr(self, "mapping", None)
        shard = getattr(draft_model, "_dflash_fc_shard", None)
        if mapping is None or mapping.tp_size <= 1 or mapping.enable_attention_dp or shard is None:
            return None
        k = int(getattr(getattr(self, "spec_config", None), "max_draft_len", 0) or 0)
        rows = tuple(
            b * (k + 1)
            for b in range(1, MAX_FAST_BATCH + 1)
            if 0 < b * (k + 1) <= TAIL_FORK_MAX_ROWS and shard.applies(draft_model.fc, b * (k + 1))
        )
        w = shard.w_shard
        if not rows or w.dtype != torch.bfloat16 or w.shape[0] % 8 != 0:
            return None
        cols = w.shape[0] // 2
        if not _allgather_cast.precompile_f32(mapping, cols, rows, w.device, agree=self._a1_all_ok):
            logger.warning(
                "M3.1 DSpark TAIL_FC_GATHER: a compile failed on some rank; the fc keeps NCCL"
            )
            return None
        return (_allgather_cast.workspace_args(mapping, cols), rows, cols)

    # -------------------------------------------------------------------------------- TAILGLUE_B / MEMCPY86K

    _decode_step = False  # this step is decode-only (TAILGLUE_B's forward tail stores it; MEMCPY86K names its block)
    _nnt_fold_target = None  # TAILGLUE_B: the model's M3DSparkStepFoldFwd (bind_tail_fold), or None
    _nnt_fold_logged = ()  # TAILGLUE_B: the (requests, captured) takes whose line this rank logged
    _tfold_inputs = None  # this step's fused drafter inputs (dspark_tailfold.StepInputs; MEMCPY86K's noise block)
    _tf86_on = False  # MEMCPY86K set up (prebuild)
    _tf86_named = False  # MEMCPY86K: this step named its noise block to the drafter
    _tf86_capt = False  # MEMCPY86K: ... while a CUDA graph is captured (the engagement line reports that path)
    _tf86_step = None  # MEMCPY86K: this decode-only step is captured (True / False), else None
    _tf86_logged = False
    _tf86_logged_eager = (
        False  # the first decode-only step run eagerly after the captured take (logged)
    )
    _tf86_logged_stock = False  # the first decode-only step captured with the copy (logged)

    def bind_tail_fold(self, fold) -> None:
        """TAILGLUE_B: the target's in-forward step fold, whose tail store the step's next_new_tokens launch can do."""
        self._nnt_fold_target = fold

    def _nnt_fold_log(self, batch: int) -> None:
        """TAILGLUE_B's engagement line: once per rank at the first take under CUDA-graph capture (the captured path),
        and one line per (requests, eager / captured) take."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        captured = torch.cuda.is_current_stream_capturing()
        if captured and not any(c for _, c in self._nnt_fold_logged):
            logger.info(
                f"[M3 dispatch] TAILGLUE_B taken: next_new_tokens + forward-tail fold in one launch ({batch} "
                f"requests, under CUDA-graph capture) on rank {rank}"
            )
        key = (batch, captured)
        if key not in self._nnt_fold_logged:
            self._nnt_fold_logged = (*self._nnt_fold_logged, key)
            logger.info(
                f"[M3 sm107 tp2] TAILGLUE_B at {batch} requests ({'captured' if captured else 'eager'}): "
                f"next_new_tokens + forward-tail fold in one launch on rank {rank}"
            )

    def _tfold_hook_inputs(self, inputs_fn):
        """The DRAFTER_INPUTS_FUSED hook, recording its outputs per step when MEMCPY86K uses them."""
        if not _tfold.MEMCPY86K:
            return inputs_fn
        return _tfold.recording_inputs(self, inputs_fn)

    def _tfold_end_step(self, draft_model) -> None:
        """Step end: MEMCPY86K's engagement read from the drafter (and its name of the block dropped)."""
        self._tfold_inputs = None
        step, self._tf86_step = self._tf86_step, None
        taken = bool(
            self._tf86_named
            and draft_model is not None
            and getattr(draft_model, "_dflash_owned_noise_taken", False)
        )
        if self._tf86_named and draft_model is not None:
            if taken:
                m3_report.work(
                    "MEMCPY86K", "drafter residual stream in the DRAFTER_INPUTS_FUSED noise block"
                )
            if taken and self._tf86_capt and not self._tf86_logged:
                self._tf86_logged = True
                logger.info(
                    f"[M3 dispatch] MEMCPY86K taken: drafter residual stream starts in the fused inputs' noise "
                    f"block, no copy on rank {getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
                )
            draft_model._dflash_owned_noise = None
            draft_model._dflash_owned_noise_taken = False
        if step is not None:
            self._tf86_note(
                step,
                None
                if taken
                else (
                    "the drafter forward did not take the named block"
                    if self._tf86_named
                    else "the step's noise block is not the DRAFTER_INPUTS_FUSED one"
                ),
            )
        self._tf86_named = False
        self._tf86_capt = False

    def _tf86_note(self, capt: bool, why: Optional[str]) -> None:
        """MEMCPY86K's two lines beside the captured take's (once per rank each) for a decode-only step: the first one
        run eagerly after the captured take, and the first one captured with the copy. ``why`` is the reason the
        drafter's residual stream did not start in the noise block, None when it did."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        if capt and why is not None and not self._tf86_logged_stock:
            self._tf86_logged_stock = True
            logger.info(
                f"[M3 dispatch] MEMCPY86K stock path while a CUDA graph is captured ({why}) on rank {rank}"
            )
        elif not capt and self._tf86_logged and not self._tf86_logged_eager:
            self._tf86_logged_eager = True
            what = "taken" if why is None else f"stock path ({why})"
            logger.info(
                f"[M3 dispatch] MEMCPY86K at an eager step after the capture: {what} on rank {rank}"
            )

    def _forward_impl(self, *args, **kwargs):
        """The S4 / DFlash step; a forked step's aux stream is always joined and kept captures are always written
        before the step returns, and no step state (TAIL_FORK, TAIL_DEFER_CAPTURE, TAILGLUE_B, MEMCPY86K) outlives
        the step."""
        attn = kwargs.get("attn_metadata", args[4] if len(args) > 4 else None)
        self._decode_step = attn is not None and int(getattr(attn, "num_contexts", 1)) == 0
        done = False
        try:
            out = super()._forward_impl(*args, **kwargs)
            done = True
            return out
        finally:
            self._decode_step = False
            st = self._tf
            if st is not None and st.forked:
                self._tf_join(st)
                if done:
                    self._tf_account(st)
            dc = self._dc
            if dc is not None:
                self._dc_flush(dc)
            self._tf = None
            self._dc = None
            self._tfold_end_step(kwargs.get("draft_model", args[6] if len(args) > 6 else None))

    # ------------------------------------------------------------------------------------------- TAIL_FORCEVEC

    _tfv_warm = False  # TAIL_FORCEVEC: the draw kernel compiled (prebuild)
    _tfv_logged = ()  # the (requests, contexts) shapes whose engagement line this rank logged

    def _apply_force_accepted_tokens(
        self, num_accepted_tokens, num_contexts, runtime_draft_len: int, spec_metadata=None
    ):
        """TAIL_FORCEVEC: the in-tree fractional forced acceptance in one launch (m3_force_vec), else the in-tree
        lines."""
        if TAIL_FORCEVEC and self._tfv_warm and self.force_num_accepted_tokens != 0.0:
            done = self._tfv_draw(
                num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata
            )
            if done is not None:
                return done
        return super()._apply_force_accepted_tokens(
            num_accepted_tokens, num_contexts, runtime_draft_len, spec_metadata=spec_metadata
        )

    def _tfv_draw(self, num_accepted_tokens, num_contexts, runtime_draft_len: int, spec_metadata):
        """The draw kernel's call where it reproduces the in-tree draw path (fractional part > 0 and a base count below
        the maximum: one counter bump, then every generation request's count), else None."""
        if spec_metadata is not None and (
            spec_metadata.is_cuda_graph and not torch.cuda.is_current_stream_capturing()
        ):
            return None  # the warmup: the in-tree method returns unchanged counts
        force = self.force_num_accepted_tokens
        int_part = int(force)
        frac = force - int_part
        max_total = runtime_draft_len + 1
        base_total = min(int_part + 1, max_total)
        if not (frac > 0.0 and base_total < max_total):
            return None
        pool = getattr(self, "_force_accept_rng_pool", None)
        counter = getattr(self, "_force_accept_rng_counter", None)
        if pool is None or counter is None:
            return None
        num_gens = num_accepted_tokens.shape[0] - num_contexts
        if (
            num_accepted_tokens.dim() != 1
            or num_accepted_tokens.dtype != torch.int32
            or not num_accepted_tokens.is_contiguous()
            or num_gens < 1
            or num_gens > 64
        ):
            return None
        m3_force_vec(
            pool,
            counter,
            num_accepted_tokens[num_contexts:],
            num_gens,
            base_total,
            max_total,
            float(frac),
            _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
            _FORCE_ACCEPT_RNG_SLOT_STRIDE,
            _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
            max(16, 1 << (num_gens - 1).bit_length()),
        )
        key = (num_gens, int(num_contexts))
        if key not in self._tfv_logged:
            self._tfv_logged = (*self._tfv_logged, key)
            logger.info(
                f"[M3 sm107 tp2] fast path engaged: TAIL_FORCEVEC taken ({num_gens} requests"
                f"{f', {int(num_contexts)} contexts' if num_contexts else ''}): forced accepted counts in one launch "
                f"on rank {getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
            )
        m3_report.work("TAIL_FORCEVEC", f"{num_gens} requests, {int(num_contexts)} contexts")
        return num_accepted_tokens

    def _tfv_prebuild(self, device) -> bool:
        """Build the forced-acceptance pool (the in-tree state, counter at zero) and compile the draw kernel for every
        block width, outside capture; a scratch counter keeps the real draw sequence untouched."""
        force = getattr(self, "force_num_accepted_tokens", 0.0)
        if not TAIL_FORCEVEC or force - int(force) <= 0.0:
            return False
        self._ensure_force_accept_rng_state(device)
        scratch = torch.zeros(1, dtype=torch.int64, device=device)
        dummy = torch.empty(64, dtype=torch.int32, device=device)
        for block in (16, 32, 64):
            m3_force_vec(
                self._force_accept_rng_pool,
                scratch,
                dummy,
                block,
                1,
                2,
                0.5,
                _FORCE_ACCEPT_RNG_COUNTER_STRIDE,
                _FORCE_ACCEPT_RNG_SLOT_STRIDE,
                _FORCE_ACCEPT_RNG_POOL_SIZE - 1,
                block,
            )
        torch.cuda.synchronize(device)
        self._tfv_warm = True
        return True

    def tail_eligible(self, attn_metadata, spec_metadata, lm_head: nn.Module) -> bool:
        """Never: the target logits always go through the logits processor and the in-tree strict accept."""
        return False

    def prebuild(self, lm_head: nn.Module, draft_model) -> None:
        """The one-launch Markov chain's shards, buffers, exchange workspace and compiles (outside capture, every
        rank); no fused verify, no coded target head, no fused draft loop."""
        rank = getattr(getattr(self, "mapping", None), "tp_rank", None)
        missing = [name for name in _WORKER_HOOKS if not hasattr(self, name)]
        if missing:
            logger.info(
                f"M3.1 DSpark: prebuild skipped on rank {rank}: the DSpark worker has no {missing[0]} hook "
                f"({len(missing)} of {len(_WORKER_HOOKS)} missing); the in-tree DSpark worker paths run"
            )
            return None
        # S10: the drafter's strided-view glue (bitwise; kill switch TRTLLM_M31_DISABLE_DRAFTER_LEAN_GLUE, read at
        # import).
        k = getattr(getattr(self, "spec_config", None), "max_draft_len", 0) or 0
        lean = not m31_disabled("DRAFTER_LEAN_GLUE") and s10_applies(k)
        self.enable_lean_glue(draft_model, lean)
        # DRAFT_WINDOW_CAPPED_DECODE (opt-in): window-capped max_seq_len for the drafter's sliding-window decode (not
        # bitwise: the TRTLLM-Gen tile / KV split changes, not the keys read).
        s5_max_batch = s5_max_batch_for(k) if m31_enabled("DRAFT_WINDOW_CAPPED_DECODE") else 0
        self.enable_window_capped_decode(draft_model, s5_max_batch)
        self.enable_full_slot_arena()
        # DRAFT_FC_TP_SPLIT (opt-in): the drafter's replicated fc split row-parallel with one fp32 all-reduce (not
        # bitwise: the partial-sum order changes). Collective.
        s2 = self.enable_fc_tp_split(draft_model, m31_enabled("DRAFT_FC_TP_SPLIT"))
        split = getattr(draft_model, "_dflash_fc_split", None)
        # DRAFT_FC_SHARD (opt-in; bitwise by its boot self-check at each of its row counts, the draft-length verify rows
        # of batches 1..MAX_FAST_BATCH within its bound; collective when on: every rank builds or every rank declines).
        # DRAFT_FC_TP_SPLIT, when on, wins. No drafter (a shell worker): nothing to set.
        if draft_model is not None and getattr(draft_model, "fc", None) is not None:
            fc_rows = tuple(
                b * (k + 1)
                for b in range(1, MAX_FAST_BATCH + 1)
                if k > 0 and b * (k + 1) <= TAIL_FORK_MAX_ROWS
            )
            draft_model._dflash_fc_shard = (
                FcShard.build(draft_model.fc, self.mapping, rows=fc_rows)
                if FC_SHARD16 and not s2 and fc_rows
                else None
            )
        logger.info(
            f"M3.1 DSpark DRAFT_FC_SHARD: drafter fc as column shards + all-gather at up to {TAIL_FORK_MAX_ROWS} rows "
            f"{'on' if getattr(draft_model, '_dflash_fc_shard', None) is not None else 'off'} on rank {rank}"
            + (
                ""
                if FC_SHARD16
                else " (default off: opt in with TRTLLM_M31_ENABLE_DRAFT_FC_SHARD=1)"
            )
        )
        ar = split[3] if split is not None else None
        ar_path = (
            "MNNVL"
            if getattr(ar, "mnnvl_allreduce", None) is not None
            else getattr(ar, "strategy", None)
        )
        logger.info(f"M3.1 DSpark DRAFT_FC_TP_SPLIT: fc TP split {'on' if s2 else 'off'} on rank {rank} (all-reduce "
                    f"{ar_path})")  # fmt: skip
        logger.info(f"M3.1 DSpark DRAFT_WINDOW_CAPPED_DECODE: window-capped draft decode for batches <= "
                    f"{s5_max_batch} on rank {rank}")  # fmt: skip
        # DRAFT_FMHA_BATCH2_SPLIT (opt-in; dspark_draft_fmha): the batch-2 window decode as two per-request
        # window-capped calls on two streams (not bitwise: drafter-only precision). Config only, the same on every
        # rank.
        from . import dspark_draft_fmha as _dfmha

        _, dfmha_line = (
            _dfmha.install(draft_model, rank) if k > 0 else (False, "off: no draft length")
        )
        logger.info(f"M3.1 DSpark DRAFT_FMHA_BATCH2_SPLIT: {dfmha_line} on rank {rank}")
        logger.info(f"M3.1 DSpark S10: lean drafter glue {'on' if lean else 'off'} on rank {rank}")
        # S9b: each drafter layer's all-reduce fused with the add + RMSNorm after it (not bitwise: the sum is not
        # rounded before the add). On by default; kill switch TRTLLM_M31_DISABLE_DRAFT_AR_NORM_FUSED, read at import.
        s9b = self.enable_fused_ar_norm(draft_model, not m31_disabled("DRAFT_AR_NORM_FUSED"))
        logger.info(
            f"M3.1 DSpark S9b: fused drafter all-reduce + norm {'on' if s9b else 'off'} on rank {rank}"
        )
        # S9B-MC: the multicast workspace is the target's (created collectively at its first eager boundary, the
        # warmup forward, before any capture); until it exists the op declines on every rank alike.
        installed = self.enable_fused_ar_norm_op(
            draft_model,
            _s9b_count_op(self.mapping, _s9b_mc_op(self.mapping) if S9B_MC else None)
            if s9b
            else None,
        )
        s9b_mc = bool(installed) and S9B_MC
        logger.info(
            f"M3.1 DSpark S9B-MC: drafter all-reduce + norm on the multicast kernel (<= {S9B_MC_MAX_ROWS} rows) "
            f"{'on' if s9b_mc else 'off'} on rank {rank}"
        )
        # DRAFTER_CTX_KV_STORE_FUSED: the drafter's context K / V epilogue + accept mask + paged store in one launch
        # after the K / V GEMM (bitwise). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all of S10c) /
        # TRTLLM_M31_DISABLE_DRAFTER_CTX_KV_STORE_FUSED, read at import. TAILBK_SPEC F: the store kernel also runs the
        # k_norm weight multiply (bitwise torch's bf16 multiply; one launch fewer per step). Kill switch
        # TRTLLM_M31_DISABLE_TAIL_BOOKKEEP_SPEC, read at import.
        s10c_a_on = not (
            m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_CTX_KV_STORE_FUSED")
        )
        tailbk_f = fold_k_norm_weight(s10c_a_on and TAILBK_SPEC)
        s10c_a = self.enable_fused_ctx_store(
            _counted(
                "DRAFTER_CTX_KV_STORE_FUSED",
                self._tf_ctx_store if TAIL_FORK_KV else fused_ctx_store,
                ("TAILBK_SPEC", tailbk_f),
            )
            if s10c_a_on
            else None
        )
        if not s10c_a:
            tailbk_f = fold_k_norm_weight(False)
        # TAIL_FORK_KV preps the store's K / V only where the store wrapper that can use the prep is installed.
        self._tf_kv = bool(s10c_a) and TAIL_FORK_KV
        logger.info(
            f"M3.1 DSpark fused drafter ctx store {'on' if s10c_a else 'off'}; TAILBK_SPEC F k_norm weight "
            f"in the store kernel {'on' if tailbk_f else 'off'} on rank {rank}"
        )
        # DRAFTER_INPUTS_FUSED: the drafter's step inputs (bonus, positions, noise block) in one launch (bitwise:
        # integers and row copies). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all of S10c) /
        # TRTLLM_M31_DISABLE_DRAFTER_INPUTS_FUSED, read at import.
        s10c_c = self.enable_fused_drafter_inputs(
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_INPUTS_FUSED")
            else self._tfold_hook_inputs(_counted("DRAFTER_INPUTS_FUSED", s10c_c_drafter_inputs))
        )
        logger.info(
            f"M3.1 DSpark DRAFTER_INPUTS_FUSED: fused drafter step inputs {'on' if s10c_c else 'off'} on rank {rank}"
        )
        # DRAFTER_CTX_ADVANCE_FUSED: the drafter's context advance + read view in one launch (bitwise, integer). Kill
        # switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all of S10c) / TRTLLM_M31_DISABLE_DRAFTER_CTX_ADVANCE_FUSED,
        # read at import.
        s10c_d = self.enable_fused_ctx_advance(
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_CTX_ADVANCE_FUSED")
            else _counted("DRAFTER_CTX_ADVANCE_FUSED", advance_ctx_and_read_view)
        )
        logger.info(
            f"M3.1 DSpark fused drafter context advance {'on' if s10c_d else 'off'} on rank {rank}"
        )
        # DRAFTER_BLOCK_TABLE_REFRESH_FUSED: the drafter's block-table refresh (decode + placeholder count) in one
        # launch (bitwise, integer). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all of S10c) /
        # TRTLLM_M31_DISABLE_DRAFTER_BLOCK_TABLE_REFRESH_FUSED, read at import. DRAFTER_BLOCK_TABLE_REFRESH_GRID: that
        # refresh on a (row, column-chunk) grid (bitwise; flat at short tables, removes the serial kernel's growth past
        # ~11K blocks per sequence). Kill switch TRTLLM_M31_DISABLE_DRAFTER_BLOCK_TABLE_REFRESH_GRID, read at import.
        s10c_b_on = not (
            m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_BLOCK_TABLE_REFRESH_FUSED")
        )
        s10c_b2 = enable_parallel_refresh(
            s10c_b_on and not m31_disabled("DRAFTER_BLOCK_TABLE_REFRESH_GRID")
        )
        s10c_b = self.enable_fused_block_refresh(
            _counted(
                "DRAFTER_BLOCK_TABLE_REFRESH_FUSED", refresh_block_tables, ("S10C_B2", s10c_b2)
            )
            if s10c_b_on
            else None
        )
        if not s10c_b:
            s10c_b2 = enable_parallel_refresh(False)
        logger.info(
            f"M3.1 DSpark fused drafter block-table refresh {'on' if s10c_b else 'off'} on rank {rank}"
            f" (DRAFTER_BLOCK_TABLE_REFRESH_GRID 2-D grid {'on' if s10c_b2 else 'off'})"
        )
        # DRAFTER_FORWARD_INDICES_FUSED: the draft model's paged block-decode index prelude in one launch (bitwise,
        # integer). Kill switches TRTLLM_M31_DISABLE_DRAFTER_FUSED_GLUE (all of S10c) /
        # TRTLLM_M31_DISABLE_DRAFTER_FORWARD_INDICES_FUSED, read at import.
        s10c_e = self.enable_fused_forward_indices(
            draft_model,
            None
            if m31_disabled("DRAFTER_FUSED_GLUE") or m31_disabled("DRAFTER_FORWARD_INDICES_FUSED")
            else _counted("DRAFTER_FORWARD_INDICES_FUSED", forward_indices),
        )
        logger.info(
            f"M3.1 DSpark fused drafter forward indices {'on' if s10c_e else 'off'} on rank {rank}"
        )
        # EX11: the drafter block's q / k RMSNorm + partial RoPE in one launch (bitwise to flashinfer's three). Kill
        # switch TRTLLM_M31_DISABLE_EX11, read at import.
        ex11 = self.enable_fused_qk_norm_rope(
            draft_model, None if m31_disabled("EX11") else _counted("EX11", qk_norm_rope)
        )
        logger.info(
            f"M3.1 DSpark EX11: fused drafter q/k norm + RoPE {'on' if ex11 else 'off'} on rank {rank}"
        )
        swiglu = install_draft_swiglu(draft_model)
        logger.info(f"M3.1 DSpark DRAFT_SWIGLU: {native_swiglu.engagement(swiglu)} on rank {rank}")
        # DRAFT_APPEND_FUSE: on top of EX11, the same launch also writes the block's K / V into the layer's paged
        # cache (bitwise flashinfer's append), and the per-layer append launch is skipped. Kill switch
        # TRTLLM_M31_DISABLE_DRAFT_APPEND_FUSE, read at import.
        append_fuse = self.enable_fused_qk_norm_rope_append(
            draft_model, _qk_norm_rope_append_logged if ex11 and DRAFT_APPEND_FUSE else None
        )
        logger.info(
            f"M3.1 DSpark DRAFT_APPEND_FUSE: drafter K / V append in the EX11 launch {'on' if append_fuse else 'off'} "
            f"on rank {rank}"
            + ("" if DRAFT_APPEND_FUSE else " (TRTLLM_M31_DISABLE_DRAFT_APPEND_FUSE set)")
        )
        # MXDRAFT: the drafter's qkv / o / gate_up / down GEMMs on this target's MXFP8 GEMV for the decode passes'
        # rows (B x K, B <= MAX_FAST_BATCH, up to the GEMV's own row bound), from the exact MXFP8 form of the shipped
        # weights (drafter-only precision, gated on natural AL). Kill switch TRTLLM_M31_DISABLE_DRAFTER_MXFP8, read
        # at import.
        gvc = (0, 0)
        if DRAFTER_HISLAYOUT and k > 0 and draft_model is not None:
            gvc = route_projection_layout(
                draft_model, [b * k for b in range(1, MAX_FAST_BATCH + 1)]
            )
        logger.info(
            f"[M31_TP2_GVC] DRAFTER_HISLAYOUT {'on' if DRAFTER_HISLAYOUT else 'off'}: registered {gvc[0]} projections "
            f"(fusedMLP OFF, current TP2 tactics, ROW forward wrapped) on rank {rank}; natural AL gate required"
        )
        mx = (
            0
            if not MXDRAFT or k <= 0
            else route_mx_linears(draft_model, [b * k for b in range(1, MAX_FAST_BATCH + 1)])
        )
        logger.info(f"M3.1 DSpark MXDRAFT: MXFP8 drafter GEMV on {mx} projections on rank {rank}")
        # DRAFT_LOWM_GEMM (opt-in): the decode-block GEMMs the MXFP8 GEMV does not execute on the pinned low-M kernels
        # (compiled here, outside capture).
        t6 = (
            0
            if not T6 or k <= 0
            else route_lowm_linears(
                draft_model,
                [b * k for b in range(1, MAX_FAST_BATCH + 1)],
                handoff=MXDRAFT_RP,
                rank=rank,
            )
        )
        logger.info(
            f"M3.1 DSpark DRAFT_LOWM_GEMM: low-M drafter GEMMs on {t6} projections on rank {rank} ("
            + (
                "default off: opt in with TRTLLM_M31_ENABLE_DRAFT_LOWM_GEMM=1"
                if not T6
                else f"MXFP8 hand-off {'on' if MXDRAFT_RP else 'off'}"
            )
            + ")"
        )
        if k > 0:
            logger.info(routes_line(draft_model, (k, 2 * k), rank))
        # DRAFT_MLP_MXFP8_14_ROWS (opt-in; dspark_draft_mlp): the drafter MLP's gate_up + SwiGLU at the draft-length-7
        # batch-2 block (14 rows) as MXFP8 quantize + MX GEMV with the SwiGLU in its epilogue (drafter-only precision;
        # the routes line above keeps the default names). Collective agreement when on.
        from . import dspark_draft_mlp as _dmlp

        logger.info(_dmlp.prebuild(draft_model, k, rank, self._a1_all_ok))
        self.prebuild_s4(lm_head, draft_model)
        # DRAFT_HEAD_NVFP4 (dspark_draft_head_fp4) first, then DRAFT_HEAD_MXFP8 unless the NVFP4 copy took the drafter
        # head.
        self._prebuild_draft_heads(lm_head, k, rank, draft_model)
        # DHEAD_NOSLICE: the MARKOV16 fold kernels for the head copy's padded output window (collective when a head copy
        # that declares padded_window is set up: it is on every rank or on none); its off line otherwise.
        self.prebuild_dhead_noslice()
        # N3 A2 compiles (rank-local), then the N3 A1 workspace and compiles (collective), outside capture.
        if not N3A_DISABLED and lm_head is not None:
            _strict_accept.precompile((self.max_draft_len,), lm_head.weight.device)
        self._a1 = self._prebuild_a1(lm_head) if lm_head is not None else None
        if DRAFT_A1 and self._a1 is not None:
            self._prebuild_draft_a1(lm_head, draft_model)
        # The rank-agreed Markov-chain set-up and its logits switch decide whether any vocab-sharded draft logits reach
        # the gather DRAFT_LOGITS_ALLGATHER_MC replaces; when none can, it is not compiled and its line says why (every
        # rank alike).
        mcag_off = mcag_unreachable(self) if self._a1 is not None and MCAG else None
        self._mcag_cols = (
            self._prebuild_mcag(lm_head)
            if self._a1 is not None and MCAG and mcag_off is None
            else 0
        )
        mcag = self.enable_draft_logits_gather_op(self._mcag_gather if self._mcag_cols else None)
        logger.info(
            "M3.1 DRAFT_LOGITS_ALLGATHER_MC: draft-logits all-gather on the multicast kernel "
            f"{'on' if mcag else 'off'} on rank {rank}" + (f" ({mcag_off})" if mcag_off else "")
        )
        n3a_kv = False
        if not N3A_KV_DISABLED and lm_head is not None:
            _strict_accept.precompile_kv_rewind(lm_head.weight.device)
            self._n3a_kv_buf = torch.empty(
                _strict_accept.max_requests(), dtype=torch.int32, device=lm_head.weight.device
            )
            n3a_kv = self.enable_fused_kv_rewind(self._n3a_kv_rewind)
        logger.info(
            f"M3.1 N3A-KV: one-launch draft-forward KV bookkeeping {'on' if n3a_kv else 'off'} on rank "
            f"{getattr(getattr(self, 'mapping', None), 'tp_rank', None)}"
        )
        # TAIL_FORCEVEC: the forced-acceptance pool and the draw kernel's compiles (rank-local, outside capture).
        warm = (
            self._tfv_prebuild(lm_head.weight.device)
            if lm_head is not None and getattr(lm_head, "weight", None) is not None
            else False
        )
        logger.info(
            f"M3.1 DSpark TAIL_FORCEVEC: one-launch forced counts {'on' if warm else 'off'} on rank {rank}"
        )
        # TAILGLUE_B: the one-launch next_new_tokens + fold tail (rank-local compiles, outside capture).
        nnt_fold = TAILGLUE_B and self._nnt_fold_target is not None and lm_head is not None
        if nnt_fold:
            m3_nnt_fold.precompile(lm_head.weight.device, (torch.int32, torch.int64))
        logger.info(
            f"M3.1 DSpark TAILGLUE_B: next_new_tokens + forward-tail fold in one launch {'on' if nnt_fold else 'off'} "
            f"on rank {rank}"
            + (
                ""
                if nnt_fold
                else (
                    " (TRTLLM_M31_DISABLE_TAILGLUE_B / TRTLLM_M31_DISABLE_STRICT_ACCEPT_FUSED set)"
                    if not TAILGLUE_B
                    else " (no forward-tail fold bound: TAIL_DSFOLD_FWD off)"
                )
            )
        )
        # MEMCPY86K: the drafter's residual stream starts in the DRAFTER_INPUTS_FUSED noise block (needs
        # DRAFTER_INPUTS_FUSED and the drafter's owned-noise hook).
        self._tf86_on = (
            _tfold.MEMCPY86K and bool(s10c_c) and _tfold.drafter_owns_noise_hook(draft_model)
        )
        logger.info(
            _tfold.memcpy86k_setup_line(
                rank,
                self._tf86_on,
                "TRTLLM_M31_DISABLE_MEMCPY86K set"
                if not _tfold.MEMCPY86K
                else (
                    "DRAFTER_INPUTS_FUSED off"
                    if not s10c_c
                    else "the drafter has no owned-noise hook"
                ),
            )
        )
        # TAIL_DEFER_CAPTURE: the MC finalize boundary's capture residuals kept by reference on decode-only steps.
        self._dc_on = TAIL_DEFER_CAPTURE
        logger.info(
            f"M3.1 DSpark TAIL_DEFER_CAPTURE: MC finalize capture residuals kept by reference and written before the "
            f"drafter reads them {'on' if self._dc_on else 'off'} on rank {rank}"
        )
        # TAIL_FORK: the fork's stream, then (DRAFT_FC_SHARD on) TAIL_FC_GATHER's workspace after A1's (collective).
        self._tf_prebuild(draft_model)

    def _prebuild_draft_heads(self, lm_head: nn.Module, k: int, rank, draft_model=None) -> None:
        """The drafter head's copies, by precedence: DRAFT_HEAD_NVFP4 (dspark_draft_head_fp4: an NVFP4 copy whose
        activation scale comes from the drafter's final RMSNorm; opt-in, collective agreement when on) first; when every
        rank set it up it serves the head and DRAFT_HEAD_MXFP8 is not built (one line names why); otherwise
        DRAFT_HEAD_MXFP8 (an MXFP8 copy; opt-in, collective agreement when on)."""
        from . import dspark_draft_head_fp4 as _dheadfp4

        fp4_on, fp4_line = _dheadfp4.prebuild(
            getattr(self, "_s4", None),
            lm_head,
            k,
            rank,
            getattr(getattr(self, "mapping", None), "tp_size", None),
            self._a1_all_ok,
            norm=getattr(getattr(draft_model, "model", None), "norm", None),
        )
        logger.info(fp4_line)
        if fp4_on:
            why = (
                "DRAFT_HEAD_NVFP4 set up: the NVFP4 copy takes precedence"
                if DHEAD_MX8
                else "default off: opt in with TRTLLM_M31_ENABLE_DRAFT_HEAD_MXFP8=1"
            )
            # "not built", not "MXFP8 draft head off": the latter names a DRAFT_HEAD_MXFP8 set-up failure.
            logger.info(
                f"M3.1 DSpark DRAFT_HEAD_MXFP8: not built on rank {rank} ({why}); NVFP4 draft head"
            )
            return
        self._prebuild_dhead8(lm_head)

    def _prebuild_dhead8(self, lm_head: nn.Module) -> bool:
        """Set up DRAFT_HEAD_MXFP8 on the Markov-chain state (every rank or none); False, with one line naming why, when
        it is off.
        Off by its switch: no collective (the environment is the same on every rank). On: every rank builds the copy
        (or names why not) and the ranks agree before any of them uses it."""
        mapping = getattr(self, "mapping", None)
        rank = getattr(mapping, "tp_rank", None)
        if not DHEAD_MX8:
            why = "default off: opt in with TRTLLM_M31_ENABLE_DRAFT_HEAD_MXFP8=1"
            logger.info(
                f"M3.1 DSpark DRAFT_HEAD_MXFP8: MXFP8 draft head off on rank {rank} ({why}); BF16 draft head"
            )
            return False
        state = getattr(self, "_s4", None)
        k = int(getattr(getattr(self, "spec_config", None), "max_draft_len", 0) or 0)
        why = _dhead8.unfit(lm_head, getattr(state, "shard", None), k)
        head8 = None
        if why is None:
            try:
                head8 = _dhead8.DraftHeadMx8(lm_head, [b * k for b in _dhead8.BATCHES], rank=rank)
            except Exception as e:  # noqa: BLE001 - any set-up failure (a FlashInfer compile, memory) declines on
                # every rank through the agreement below instead of leaving the ranks on different heads
                why = f"set-up failed: {type(e).__name__}: {e}"
        if not self._a1_all_ok(why is None):
            logger.info(
                f"M3.1 DSpark DRAFT_HEAD_MXFP8: MXFP8 draft head off on rank {rank} ({why or 'off on another rank'}); "
                "BF16 draft head"
            )
            return False
        state.head8 = head8
        logger.info(
            f"M3.1 DSpark DRAFT_HEAD_MXFP8: MXFP8 draft head set up on rank {rank}/{getattr(mapping, 'tp_size', None)} "
            f"(shard {head8.n} rows padded to {head8.np}, K {head8.k}, rows {head8.rows}, tactic {_dhead8.TACTIC}, "
            f"quantize {head8.quantizer}, copy {head8.bytes / 1e6:.1f} MB, set-up {head8.setup_s:.2f} s)"
        )
        return True

    _n3a_kv_buf = None  # the persistent rewind amounts of N3A-KV (int32, MAX_REQUESTS)

    def _n3a_kv_rewind(self, num_accepted, kv_lens, nc: int, bs: int):
        """N3A-KV for the DFlash hook: ``1 - num_accepted[nc:bs]`` (a view of the persistent buffer) with
        ``kv_lens[nc:bs] += 1`` in one launch, or None (the stock lines) for operands or a variant it does not take."""
        n = bs - nc
        buf = self._n3a_kv_buf
        if buf is None or not _strict_accept.supports_kv_rewind(num_accepted, kv_lens, buf, nc, n):
            return None
        if not _strict_accept.is_compiled_kv_rewind(num_accepted, kv_lens):
            return None
        _n3a_kv_log_once()
        m3_report.work("N3A_KV", f"{n} requests")
        return _strict_accept.spec_kv_rewind(num_accepted, kv_lens, buf, nc, n)

    _a1 = None  # the N3 A1 workspace (allgather_cast's arguments) once set up

    def _a1_all_ok(self, local: bool) -> bool:
        """Whether ``local`` holds on every rank of the TP group: A1 is set up on every rank or on none (a rank that
        is not eligible, or whose compile failed, turns it off everywhere)."""
        from tensorrt_llm._torch.distributed.ops import (
            _get_mnnvl_workspace_comm,
            _mnnvl_workspace_all_succeeded,
        )

        return _mnnvl_workspace_all_succeeded(_get_mnnvl_workspace_comm(self.mapping), local)

    def _prebuild_a1(self, lm_head: nn.Module):
        """The N3 A1 workspace and its K3 verify-row compiles, or None (every rank alike): off by its switch, at
        another draft length, without a column-parallel bf16 head over a TP group, or when a compile fails on any
        rank."""
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        mapping = self.mapping
        if mapping.tp_size <= 1 or mapping.enable_attention_dp:
            return None
        cols = int(getattr(lm_head, "out_features", 0))
        sms = torch.cuda.get_device_properties(lm_head.weight.device).multi_processor_count
        # Why this rank would not set A1 up (empty: it would). The grid is a function of rows and the SM count:
        # requiring every SM up to the workspace's MAX_GRID makes it a function of rows alone, the same on every rank.
        off = [
            why
            for why, bad in (
                (
                    "TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC set",
                    A1_DISABLED,
                ),
                (
                    f"draft length {int(self.max_draft_len)} with "
                    "TRTLLM_M31_DISABLE_VERIFY_LOGITS_ALLGATHER_MC_ANY_DRAFT_LEN set",
                    int(self.max_draft_len) != 3 and A1_K7_DISABLED,
                ),
                (
                    f"draft length {int(self.max_draft_len)}: more than {A1_MAX_ROWS} verify rows at batch 1",
                    int(self.max_draft_len) + 1 > A1_MAX_ROWS,
                ),
                (
                    "not a column-parallel, gathered, unpadded, bias-free head",
                    getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN
                    or not getattr(lm_head, "gather_output", False)
                    or getattr(lm_head, "padding_size", 0) != 0
                    or getattr(lm_head, "bias", None) is not None,
                ),
                (f"head dtype {lm_head.weight.dtype}", lm_head.weight.dtype != torch.bfloat16),
                (
                    f"{cols} shard columns (whole 16-byte vectors needed)",
                    cols <= 0 or cols % 8 != 0,
                ),
                (
                    f"{sms} SMs (fewer than {_allgather_cast.MAX_GRID})",
                    sms < _allgather_cast.MAX_GRID,
                ),
            )
            if bad
        ]
        if not self._a1_all_ok(not off):
            logger.info(
                f"M3.1 N3 A1: off on rank {mapping.tp_rank} ({'; '.join(off) or 'off on another rank'}); "
                "the verify logits use the NCCL all-gather"
            )
            return None
        k1 = int(self.max_draft_len) + 1
        rows = tuple(b * k1 for b in range(1, A1_MAX_ROWS // k1 + 1))
        start = time.perf_counter()
        _allgather_cast.configure_capacity(A1_MAX_ROWS)
        if not _allgather_cast.precompile(
            lm_head.mapping, cols, rows, lm_head.weight.device, agree=self._a1_all_ok
        ):
            logger.warning(
                "M3.1 N3 A1: a compile failed on some rank; every rank keeps the NCCL all-gather"
            )
            return None
        ws = _allgather_cast.workspace_args(lm_head.mapping, cols)
        # The engagement line: one per rank, at server start.
        logger.info(
            f"M3.1 N3 A1: one-launch verify all-gather set up on rank {mapping.tp_rank}/{mapping.tp_size} "
            f"(shard {cols} columns, rows {rows}, workspace {ws['ws_uc'].numel() * 4} bytes/rank, "
            f"set-up {time.perf_counter() - start:.2f} s)"
        )
        return ws

    def _prebuild_draft_a1(self, target_head: nn.Module, draft_model) -> None:
        """Reuse the verify workspace for matching draft shards, compiling B*K shapes before capture."""
        from tensorrt_llm._torch.modules.linear import TensorParallelMode
        from tensorrt_llm._torch.modules.logits_processor import LogitsProcessor

        head = draft_model.lm_head
        inner = draft_model.logits_processor
        if isinstance(inner, _DraftA1Logits):
            inner = inner.inner
        k = int(self.max_draft_len)
        cols = int(getattr(head, "out_features", 0))
        eligible = (
            type(inner) is LogitsProcessor
            and k > 0
            and head.tp_mode == TensorParallelMode.COLUMN
            and head.gather_output
            and head.gather_output_sizes is None
            and head.padding_size == 0
            and head.bias is None
            and head.weight.dtype == torch.bfloat16
            and head.weight.device == target_head.weight.device
            and cols == target_head.out_features
            and head.mapping.tp_group == target_head.mapping.tp_group
            and head.mapping.tp_rank == target_head.mapping.tp_rank
        )
        if not self._a1_all_ok(eligible):
            logger.info(
                "M3.1 DRAFT_A1: incompatible draft head or custom logits processor; keeping stock"
            )
            return
        rows = tuple(b * k for b in range(1, A1_MAX_ROWS // k + 1) if b * k > 64)
        if not rows or not _allgather_cast.precompile(
            target_head.mapping, cols, rows, head.weight.device, agree=self._a1_all_ok
        ):
            logger.warning("M3.1 DRAFT_A1: draft-shape compilation unavailable; keeping stock")
            return
        draft_model.logits_processor = _DraftA1Logits(inner, head, self._a1, rows)
        logger.info(
            f"M3.1 DRAFT_A1: draft-head all-gather set up on rank {self.mapping.tp_rank}, rows {rows}"
        )

    def _prebuild_mcag(self, lm_head: nn.Module) -> int:
        """Compile the fp32 draft-logits gather for B * K rows up to its capacity, on every rank (A1's workspace and
        agreement); its shard columns, or 0 (every rank alike) when a compile fails anywhere."""
        cols = int(getattr(lm_head, "out_features", 0))
        k = int(self.max_draft_len)
        if k <= 0 or cols % 4 != 0:
            return 0
        rows = tuple(b * k for b in range(1, _allgather_cast.MAX_ROWS_FP32 // k + 1))
        if not _allgather_cast.precompile_f32(
            lm_head.mapping, cols, rows, lm_head.weight.device, agree=self._a1_all_ok
        ):
            logger.warning(
                "M3.1 DRAFT_LOGITS_ALLGATHER_MC: a compile failed on some rank; every rank keeps the NCCL all-gather"
            )
            return 0
        return cols

    def _mcag_gather(self, logits: torch.Tensor):
        """The draft logits gathered in one launch on the A1 workspace, or None (the stock all-gather) for a shape
        that is not compiled; the decision depends on the shape only, the same on every rank."""
        rows = logits.numel() // max(int(logits.shape[-1]), 1)
        if (
            logits.dtype != torch.float32
            or int(logits.shape[-1]) != self._mcag_cols
            or not _allgather_cast.is_compiled(
                rows, self._mcag_cols, self._a1["world"], device=logits.device, fp32=True
            )
        ):
            _a1_log_once(
                ("mcag", rows, logits.dtype),
                f"[M3 dispatch] DRAFT_LOGITS_ALLGATHER_MC draft-logits all-gather not taken: {rows} rows "
                f"{logits.dtype}; NCCL all-gather",
            )
            return None
        _a1_log_once(
            "mcag",
            "[M3 dispatch] DRAFT_LOGITS_ALLGATHER_MC draft-logits all-gather taken: one launch on the A1 workspace",
        )
        m3_report.work("DRAFT_LOGITS_ALLGATHER_MC", f"{rows} draft rows, world {self._a1['world']}")
        return _allgather_cast.gather_last_dim_f32(logits.contiguous(), **self._a1)

    def _n3a_eligible(
        self, logits, draft_tokens, num_contexts: int, batch_size: int, spec_metadata
    ) -> bool:
        """Whether this step's strict acceptance takes the fused op: a decode-only, non-ragged step whose logits hold
        exactly batch_size x (D + 1) verify rows and whose drafts the op takes. Host state and shapes only, so it is
        fixed per captured graph and the same on every rank. Decided before the target tokens are sampled, so the
        sampler runs exactly once either way."""
        if N3A_DISABLED or spec_metadata is None or spec_metadata.is_ragged_verify:
            return False
        if num_contexts != 0 or batch_size <= 0 or draft_tokens.dim() != 2:
            return False
        n_req, draft_len = draft_tokens.shape
        if n_req != batch_size or draft_len <= 0 or n_req > _strict_accept.max_requests():
            return False
        rows = logits.shape[0] if logits.dim() > 1 else 1
        return rows == batch_size * (draft_len + 1) and draft_tokens.is_contiguous()

    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        """The in-tree strict acceptance with its epilogue in one launch (N3 A2): the same target tokens (the in-tree
        sampler, called once), the same accepted tokens and counts, then the in-tree forced-acceptance override (its
        RNG pool and counters advance as on an in-tree step)."""
        if not self._n3a_eligible(logits, draft_tokens, num_contexts, batch_size, spec_metadata):
            return super()._sample_and_accept_draft_tokens_base(
                logits, draft_tokens, num_contexts, batch_size, spec_metadata
            )
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        target_tokens = self._sample_tokens_for_batch(
            logits, spec_metadata, num_contexts, batch_size
        )
        if _strict_accept.supports_strict_accept(
            target_tokens, draft_tokens
        ) and _strict_accept.is_compiled_strict_accept(target_tokens, draft_tokens):
            accepted, num_accepted = _strict_accept.spec_strict_accept(target_tokens, draft_tokens)
            _n3a_log_once()
            m3_report.work("N3A", f"strict accept, {draft_tokens.shape[0]} requests")
        else:
            # The sampler already ran: finish with the in-tree epilogue (the op's reference is that code).
            accepted, num_accepted = _strict_accept.spec_strict_accept.reference(
                target_tokens, draft_tokens
            )
        num_accepted = self._apply_force_accepted_tokens(
            num_accepted, num_contexts, draft_tokens.shape[-1], spec_metadata=spec_metadata
        )
        return accepted, num_accepted

    def target_logits(self, hidden_states, attn_metadata, spec_metadata, lm_head: nn.Module):
        """The verify rows' gathered fp32 target logits (N3 P-gid + P-cat), or None for the logits processor.

        On a decode-only, non-ragged step every row is a verify row and ``gather_ids`` is ``arange(num_tokens)`` (each
        generation request extends it with the next ``1 + D`` row indices), so the rows are a slice, not a gather.
        The processor's head is the column-parallel BF16 GEMM, an all-gather along the vocab (NCCL, then a concat)
        and ``.float()``; here the same all-gather runs along dim 0 (no concat) and one strided copy writes the
        rank-major shards into the fp32 output: the same bytes, an exact widening. Shapes and host state only, so
        the decision is fixed per captured graph and the same on every rank."""
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        if (
            N3P_DISABLED
            or spec_metadata is None
            or getattr(spec_metadata, "is_ragged_verify", False)
        ):
            return None
        if attn_metadata.num_contexts != 0 or spec_metadata.gather_ids is None:
            return None
        rows = hidden_states.shape[0]
        if spec_metadata.gather_ids.shape[0] != rows or rows == 0:
            return None
        if (
            getattr(lm_head, "tp_mode", None) != TensorParallelMode.COLUMN
            or not getattr(lm_head, "gather_output", False)
            or getattr(lm_head, "padding_size", 0) != 0
            or getattr(lm_head, "bias", None) is not None
        ):
            return None
        shard = lm_head.apply_linear(hidden_states, None)
        _n3p_log_once()
        m3_report.work("N3P", f"{rows} verify rows")
        tp = lm_head.tp_size
        if tp == 1:
            return shard.float()
        # N3 A1: one launch for the all-gather, the concat and the cast. The decision depends on the rows and the
        # set-up only (the same on every rank, fixed per captured graph).
        a1 = (
            self._a1 is not None
            and rows <= A1_MAX_ROWS
            and _allgather_cast.is_compiled(rows, shard.shape[-1], tp, 0, shard.device)
        )
        # TAIL_FORK: the drafter's input work forks here, after the head GEMM; after the verify logits' all-gather when
        # TAIL_FORK_AFTER_VERIFY_GATHER (with A1) or when the fork runs TAIL_FC_GATHER's multicast launch (_tf_late);
        # TAIL_FORK_ISSUE_AFTER_VERIFY_GATHER: still after the head GEMM only (its event), its kernels issued after A1's
        # launch.
        head_done = None
        if not self._tf_late(a1):
            if TAIL_FORK_A1_FIRST and a1 and self._tf is not None:
                head_done = torch.cuda.current_stream().record_event()
            else:
                self._tf_launch()
        if self._a1 is not None:
            if a1:
                _a1_log_once(
                    "taken",
                    "[M3 dispatch] n3_a1 allgather_cast taken: verify logits gathered in one launch",
                )
                m3_report.work("A1", f"{rows} verify rows, world {tp}")
                if int(self.max_draft_len) != 3:
                    m3_report.work(
                        "A1_K7", f"draft length {int(self.max_draft_len)}, {rows} verify rows"
                    )
                out = _allgather_cast.allgather_cast(shard, **self._a1)
                # A fork not launched above (a no-op otherwise): TAIL_FORK_ISSUE_AFTER_VERIFY_GATHER's on the head
                # GEMM's event, or a late one after A1.
                self._tf_launch(
                    after=head_done,
                    point="head, issued after A1's launch"
                    if head_done is not None
                    else "A1 all-gather",
                )
                return out
            _a1_log_once(
                rows,
                f"[M3 dispatch] n3_a1 allgather_cast not taken: {rows} verify rows (compiled: verify rows up to "
                f"{A1_MAX_ROWS}); NCCL all-gather",
            )
        gathered = allgather(shard, lm_head.mapping, dim=0)
        self._tf_launch(
            point="NCCL all-gather"
        )  # a late fork (a no-op once the fork launched above)
        vocab_shard = shard.shape[-1]
        out = torch.empty((rows, tp * vocab_shard), dtype=torch.float32, device=shard.device)
        out.view(rows, tp, vocab_shard).copy_(gathered.view(tp, rows, vocab_shard).transpose(0, 1))
        return out

    def _prepare_next_new_tokens(
        self,
        accepted_tokens,
        next_draft_tokens,
        batch_indices_cuda,
        batch_size,
        num_accepted_tokens,
    ):
        """The next step's input tokens in one launch (N3 A2), else the in-tree index + concat; on a decode-only step
        whose forward tail stores the step, with that tail's store + staging in the same launch (TAILGLUE_B)."""
        batch_idx = batch_indices_cuda[:batch_size]
        fold = self._nnt_fold_target
        if (
            TAILGLUE_B
            and fold is not None
            and self._decode_step
            and next_draft_tokens.dim() == 2
            and next_draft_tokens.shape[0] == batch_size
            and accepted_tokens.dim() == 2
            and accepted_tokens.shape[0] == batch_size
        ):
            out = fold.fold_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
            if out is not None:
                self._nnt_fold_log(batch_size)
                m3_report.work("TAILGLUE_B", f"next_new_tokens + fold tail, batch {batch_size}")
                return out
        if (
            not N3A_DISABLED
            and next_draft_tokens.dim() == 2
            and next_draft_tokens.shape[0] == batch_size
            and _strict_accept.supports_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
            and _strict_accept.is_compiled_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
        ):
            m3_report.work("N3A", f"next-step input tokens, batch {batch_size}")
            return _strict_accept.spec_next_new_tokens(
                accepted_tokens, num_accepted_tokens, batch_idx, next_draft_tokens
            )
        return super()._prepare_next_new_tokens(
            accepted_tokens, next_draft_tokens, batch_indices_cuda, batch_size, num_accepted_tokens
        )


def build_dspark(
    shell, model_config: ModelConfig
) -> Tuple[ModelConfig, nn.Module, M3DSparkWorker, bool]:
    """The drafter config, the drafter and the worker of one-model standalone DSpark (mirrors
    ``SpecDecOneEngineForCausalLM`` for an external drafter)."""
    spec = model_config.spec_config
    if not is_standalone_dspark(spec):
        raise ValueError("build_dspark serves the standalone DSpark drafter only")
    draft_config = ModelConfig.from_pretrained(
        spec.speculative_model, **external_drafter_config_kwargs(model_config, spec)
    )
    # The drop is safe only because draft_config owns its QuantConfig: from_pretrained reads the drafter checkpoint
    # and takes no quant_config from the target. Never pass the target's quant_config in here.
    keep_drafter_kv_dtype(draft_config)
    draft_config.extra_attrs = model_config.extra_attrs
    use_separate = should_use_separate_draft_kv_cache(spec)
    draft = get_draft_model(model_config, draft_config, shell.lm_head, shell.model)
    # A parallel drafter (DSpark) projects its block rows through the target's logits processor, as the one-engine
    wrapped = install_draft_oneshot_allreduce(draft, DRAFT_AR_ONESHOT_MAX_ROWS)
    logger.info(
        f"[M3 prebuild] S9b-VR-a: {wrapped} drafter row-parallel all-reduce(s) take ONESHOT up to "
        f"{DRAFT_AR_ONESHOT_MAX_ROWS} rows"
        + ("" if DRAFT_AR_ONESHOT_MAX_ROWS else " (off: TRTLLM_M31_DISABLE_S9BVR)")
    )
    # construction sets it.
    draft.logits_processor = shell.logits_processor
    worker = M3DSparkWorker(spec, model_config.mapping, use_separate_draft_kv_cache=use_separate)
    worker.set_draft_model(draft)
    shell.epilogue.append(draft)
    shell.epilogue.append(worker)
    logger.info(
        f"M3.1 target DSpark: {type(worker).__name__} over {type(draft).__name__}, K={spec.max_draft_len}, "
        f"separate draft KV cache={use_separate}"
    )
    return draft_config, draft, worker, use_separate


def load_dspark_draft_weights(shell, weights, weight_mapper=None) -> None:
    """The drafter's own checkpoint, then the target's embedding and head (mirrors
    ``SpecDecOneEngineForCausalLM.load_draft_weights`` for DSpark).

    The drafter borrows the target's embedding and head, and ``M3EmbedTokens.weight`` IS the target's table, so a
    drafter checkpoint that ships either would load into a module the target owns or be shadowed by it: refused.
    """
    borrowed = sorted(
        k for k in weights.keys() if "embed_tokens." in k or k.split(".")[-2:-1] == ["lm_head"]
    )
    if borrowed:
        raise ValueError(
            f"the DSpark drafter checkpoint ships tensors this target lends it (embedding / head): {borrowed[:4]}"
        )
    shell.draft_model.load_weights(weights=weights, weight_mapper=weight_mapper)
    shell.draft_model.load_weights_from_target_model(shell)
    if shell.draft_model.lm_head is not shell.lm_head or (
        shell.draft_model.draft_model_full.model.embed_tokens is not shell.model.embed_tokens
    ):
        raise RuntimeError("the DSpark drafter does not borrow this target's embedding and head")
    own = sorted(
        n for n, _ in shell.named_parameters() if not n.startswith(("model.w.", "draft_model."))
    )
    logger.info(f"M3.1 target DSpark: shell parameters outside the core and the drafter: {own}")


def keep_drafter_kv_dtype(draft_config: ModelConfig) -> None:
    """Drop any KV quantization from the drafter's config, so its context pool is allocated in its projection dtype.

    The one-engine construction copies the target's KV algo onto the drafter; on this target that is NVFP4, which
    allocates a packed int8 pool the worker refuses to bind (it falls back to its private arena, and a reused
    prefix's drafter K/V is lost). The cached quant modes are dropped with it, as the draft KV budget does.
    """
    for quant in [draft_config.quant_config, *(draft_config.quant_config_dict or {}).values()]:
        quant.kv_cache_quant_algo = None
        quant.__dict__.pop("quant_mode", None)
        quant.__dict__.pop("layer_quant_mode", None)


def draft_kv_override_applies(spec_config) -> bool:
    """The target's FP8 draft-KV override is an Eagle3 need; a DSpark drafter keeps its BF16 context pool."""
    return not is_standalone_dspark(spec_config)


__all__ = [
    "M3DSparkWorker",
    "M3EmbedTokens",
    "build_dspark",
    "draft_kv_override_applies",
    "is_standalone_dspark",
    "keep_drafter_kv_dtype",
    "load_dspark_draft_weights",
]
