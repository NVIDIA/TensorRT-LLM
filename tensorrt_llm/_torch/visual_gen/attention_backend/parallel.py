# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Parallelism Wrappers

Wraps any attention backend with a parallelism strategy. Not a standalone
backend — compose around a real backend (VANILLA/TRTLLM/FA4/CUTEDSL).

"""

from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

if TYPE_CHECKING:
    from ..mapping import VisualGenMapping

from tensorrt_llm._torch.distributed import all_to_all_4d, all_to_all_5d

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_flash_attn_combine_import_error = None
try:
    from flash_attn.cute.interface import flash_attn_combine as _flash_attn_combine
except (ImportError, OSError) as e:
    _flash_attn_combine = None
    _flash_attn_combine_import_error = e


# -----------------------------------------------------------------------------
# Pre-/post-alltoall compute functions for split-QKV pipeline.
# Each pre-compute fn fuses GEMM + (norm + rope) + 4D-view + 5D-view + permute
# + contiguous into a single inductor compile region (matching bench
# `_pV/_pQ/_pK`). inductor fuses permute+contig into the GEMM/norm/rope
# epilogue stores → 1 fused triton kernel per V/Q/K path under torch.compile.
#
# Free functions (not methods) so dynamo can trace cleanly without `self`.
# Wrapped via `torch.compile` in `UlyssesAttention.attach_compiled_pre_compute`.
# -----------------------------------------------------------------------------

def v_pre_all_to_all_comp(to_v, x, B, Sp, P, num_kv_heads, head_dim):
    """V path: GEMM + 4D view + 5D permute + contiguous.
    Returns 5D [P, B, Sp, num_kv_heads/P, head_dim] ready for v11 alltoall."""
    v = to_v(x)
    v = v.view(B, Sp, num_kv_heads, head_dim)
    return (v.view(B, Sp, P, num_kv_heads // P, head_dim)
              .permute(2, 0, 1, 3, 4)
              .contiguous())


def q_pre_all_to_all_comp(to_q, norm_q, apply_rotary_emb_fn,
                          x, B, Sp, P, num_heads, head_dim, pe, rope_type):
    """Q path: GEMM + RMSNorm + RoPE + 4D view + 5D permute + contiguous.
    NOTE: `apply_rotary_emb_fn` is wrapped with @torch.compiler.disable in
    rope.py — under inductor compile its FMA-fused mul-add diverges by ~4 ULPs
    from eager (verified by per-stage dump). With @disable, GEMM + RMSNorm
    still get fused into the inductor graph; only the RoPE math runs eager."""
    q = to_q(x)
    if norm_q is not None:
        q = norm_q(q)
    if pe is not None:
        q = apply_rotary_emb_fn(q, pe, rope_type)
    q = q.view(B, Sp, num_heads, head_dim)
    return (q.view(B, Sp, P, num_heads // P, head_dim)
              .permute(2, 0, 1, 3, 4)
              .contiguous())


def k_pre_all_to_all_comp(to_k, norm_k, apply_rotary_emb_fn,
                          x, B, Sp, P, num_kv_heads, head_dim, k_pe, rope_type):
    """K path: GEMM + RMSNorm + RoPE + 4D view + 5D permute + contiguous.
    See note in `q_pre_all_to_all_comp` — RoPE eager-only via @disable."""
    k = to_k(x)
    if norm_k is not None:
        k = norm_k(k)
    if k_pe is not None:
        k = apply_rotary_emb_fn(k, k_pe, rope_type)
    k = k.view(B, Sp, num_kv_heads, head_dim)
    return (k.view(B, Sp, P, num_kv_heads // P, head_dim)
              .permute(2, 0, 1, 3, 4)
              .contiguous())


def post_permute_5d_to_4d(out_5d, P):
    """5D [P, B, Sp, H/P, D] → 4D [B, P*Sp, H/P, D] (block-by-rank gather).
    .contiguous() copies slot data out (layout-normalize for SDPA, decoupling
    from IPC slot lifetime). Inductor fuses permute+contig with downstream
    SDPA input prep."""
    _P, Bt, Spt, HpP, Dt = out_5d.shape
    return (out_5d.permute(1, 0, 2, 3, 4)
                  .contiguous()
                  .view(Bt, _P * Spt, HpP, Dt))


_ULYSSES_ATTN_INSTANCE_COUNT = 0


class UlyssesAttention(AttentionBackend):
    """
    Ulysses Sequence Parallelism wrapper.

    Wraps any attention backend with sequence parallelism via all-to-all.
    Not a standalone backend -- compose around a real backend (VANILLA/TRTLLM).
    Fully transparent to backend-specific kwargs: everything in ``**kwargs``
    is forwarded to the inner backend unchanged (except ``seq_len`` which is
    overridden with the post-all-to-all value).

    Architecture:
        Input:  [B, S/P, H, D] (sequence sharded across P processes)
        Step 1: All-to-All → [B, S, H/P, D] (gather sequence, shard heads)
        Step 2: Compute attention with wrapped backend (VANILLA or TRTLLM)
        Step 3: All-to-All → [B, S/P, H, D] (restore sequence sharding)
        Output: [B, S/P, H, D] (sequence sharded)

    Two modes (auto-selected via ``inner_backend.support_fused_qkv()``):
    - Unfused: 3 separate all-to-all for Q/K/V + 1 for output (4 collectives)
    - Fused: stacks Q/K/V into [B, S/P, 3, H, D], 1 fused 5D all-to-all
      + 1 for output (2 collectives total)
    """

    def __init__(
        self,
        inner_backend: AttentionBackend,
        process_group: torch.distributed.ProcessGroup,
    ):
        global _ULYSSES_ATTN_INSTANCE_COUNT
        self._dbg_idx = _ULYSSES_ATTN_INSTANCE_COUNT
        _ULYSSES_ATTN_INSTANCE_COUNT += 1

        self.inner_backend = inner_backend
        self.process_group = process_group
        self._preferred_layout = AttentionTensorLayout.NHD

        self.head_dim = inner_backend.head_dim
        self.sharded_num_heads = inner_backend.num_heads
        self.sharded_num_kv_heads = getattr(inner_backend, "num_kv_heads", self.sharded_num_heads)

        self.world_size = torch.distributed.get_world_size(group=process_group)

        self.num_heads = self.sharded_num_heads * self.world_size
        self.num_kv_heads = self.sharded_num_kv_heads * self.world_size

        # GC streams populated by `attach_pipeline_streams()` at module
        # construction time (before any traced forward). Pipeline body runs
        # eager Python under `@torch.compiler.disable(recursive=False)` so
        # plain instance attributes work (no need for ctypes-backed singleton
        # lookups or workarounds for dynamo tracing).
        self.pri_comm_stream: Optional[torch.cuda.Stream] = None
        self.gc_comp_stream: Optional[torch.cuda.Stream] = None
        self.gc_selfcopy_stream: Optional[torch.cuda.Stream] = None
        self.gc_selfcopy_handle: int = 0

        # Cached for direct cpp op call (v11 alltoall). Populated lazily on
        # first attach since process_group is required. pg.boxed() returns a
        # ScriptObject that's expensive to recreate — caching it avoids ~100us
        # per alltoall call (3 calls per attention iter).
        self._pg_boxed = None
        self._group_ranks = None

        # Compiled pre-alltoall fns (set by `attach_compiled_pre_compute`).
        # Each is its own inductor compile region — fuses GEMM+norm+rope+
        # permute+contig per V/Q/K path.
        self._v_pre_compiled = None
        self._q_pre_compiled = None
        self._k_pre_compiled = None

    def _emit_barrier(self):
        """Cross-rank barrier between alltoall slot writes. Emits an NCCL
        device-API `ncclLsaBarrierSession` release fence — pri_comm's preceding
        peer-copies and self-copy completion are fenced before the slot ring
        is reused."""
        torch.ops.trtllm.ulysses_lsa_barrier(
            self._group_ranks, self._pg_boxed)

    def _alltoall_v11_issue(self, perm: torch.Tensor, gc_self_handle: int) -> torch.Tensor:
        """Issue-only hybrid SM(self)+CE(peer) alltoall on `perm`. Uses
        `ncclMemAlloc + ncclCommWindowRegister(NCCL_WIN_COLL_SYMMETRIC)` for
        the slot ring and `ncclGetPeerDevicePointer` for the peer VA mapping.
        Caller pairs each call with `_emit_barrier()` for the cross-rank fence."""
        return torch.ops.trtllm.ulysses_alltoall_hybrid_symm(
            perm, self._group_ranks, self._pg_boxed, gc_self_handle)

    def attach_pipeline_streams(self, streams) -> None:
        """Inject pre-created GC partition streams + handles. Caller is
        responsible for ensuring `streams` is the per-device singleton from
        `UlyssesPipelineStreams.get(device_id)` — created BEFORE any
        torch.compile-traced forward runs.

        Also caches `pg_boxed` and `group_ranks` for direct cpp op calls in
        `_pre_attn_alltoall_pipeline` body (which runs eager under @disable),
        avoiding per-iter PG resolve + boxify overhead (~127us/iter)."""
        self.pri_comm_stream = streams.pri_comm_stream
        self.gc_comp_stream = streams.gc_comp_stream
        self.gc_selfcopy_stream = getattr(streams, "gc_selfcopy_stream", None)
        self.gc_selfcopy_handle = streams.gc_selfcopy_handle
        if self.process_group is not None:
            self._pg_boxed = self.process_group.boxed()
            self._group_ranks = sorted(
                torch.distributed.get_process_group_ranks(self.process_group))

    def attach_compiled_pre_compute(self) -> None:
        """torch.compile each pre-alltoall fn so inductor fuses GEMM + RMSNorm
        + RoPE + 4D-view + 5D-permute + .contiguous() into one Triton kernel
        per V/Q/K path. Each compile() return is its own compile region; they
        re-enter inductor when called from inside the @disable'd
        `_pre_attn_alltoall_pipeline` (which uses `recursive=False`)."""
        _C = lambda fn: torch.compile(fn, mode="default", dynamic=False, fullgraph=False)
        self._v_pre_compiled = _C(v_pre_all_to_all_comp)
        self._q_pre_compiled = _C(q_pre_all_to_all_comp)
        self._k_pre_compiled = _C(k_pre_all_to_all_comp)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with Ulysses sequence parallelism.

        q/k/v: [B, S/P, H, D] each.  All other arguments are forwarded
        transparently to the inner backend via ``**kwargs``.
        """
        # Catches upstream floor-division bugs (e.g. num_heads // ulysses_size when
        # num_heads % ulysses_size != 0) before they corrupt the all-to-all.
        if q.shape[2] % self.world_size != 0:
            raise ValueError(
                f"UlyssesAttention: q num_heads ({q.shape[2]}) must be divisible "
                f"by world_size ({self.world_size})."
            )
        if k.shape[2] % self.world_size != 0:
            raise ValueError(
                f"UlyssesAttention: k num_kv_heads ({k.shape[2]}) must be divisible "
                f"by world_size ({self.world_size})."
            )

        if self.inner_backend.support_fused_qkv():
            return self._forward_fused(q, k, v, **kwargs)
        return self._forward_unfused(q, k, v, **kwargs)

    def _forward_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        qkv = torch.stack([q, k, v], dim=2)
        qkv = all_to_all_5d(qkv, scatter_dim=3, gather_dim=1, process_group=self.process_group)

        B, seq_len, _, Hp, D = qkv.shape

        # Caller passed pre-A2A (sharded) seq_len; the inner backend
        # reshapes by it, so hand it the post-A2A length instead.
        kwargs["batch_size"] = batch_size
        kwargs["seq_len"] = seq_len
        kwargs["seq_len_kv"] = seq_len

        output = self.inner_backend.forward(q=qkv, k=None, v=None, **kwargs)

        return self._output_a2a(output, batch_size, seq_len)

    def _forward_unfused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = q.shape[0]
        q = all_to_all_4d(q, scatter_dim=2, gather_dim=1, process_group=self.process_group)
        k = all_to_all_4d(k, scatter_dim=2, gather_dim=1, process_group=self.process_group)
        v = all_to_all_4d(v, scatter_dim=2, gather_dim=1, process_group=self.process_group)

        seq_len_full = q.shape[1]
        kv_seq_len_full = k.shape[1]

        if self.inner_backend.preferred_layout == AttentionTensorLayout.HND:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        # Caller passed pre-A2A (sharded) seq_lens; hand the inner
        # backend the post-A2A lengths instead.
        kwargs["batch_size"] = batch_size
        kwargs["seq_len"] = seq_len_full
        kwargs["seq_len_kv"] = kv_seq_len_full

        output = self.inner_backend.forward(q=q, k=k, v=v, **kwargs)

        return self._output_a2a(output, batch_size, seq_len_full)

    # ------------------------------------------------------------------
    # Bench v11_split_hybrid_batched-style 4-stream pipeline (split QKV).
    # This method is invoked from inside the `ltx2_split_qkv_pipeline` custom_op
    # implementation (registered in distributed/ops.py), which is opaque to
    # dynamo via register_fake — so stream switches inside execute eagerly
    # without triggering graph break in the surrounding torch.compile graph.
    # ------------------------------------------------------------------
    @torch.compiler.disable(recursive=False)
    def _pre_attn_alltoall_pipeline(
        self,
        x: torch.Tensor,
        to_q: torch.nn.Module,
        to_k: torch.nn.Module,
        to_v: torch.nn.Module,
        norm_q: Optional[torch.nn.Module],
        norm_k: Optional[torch.nn.Module],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        pe: Optional[tuple],
        k_pe: Optional[tuple],
        rope_type,
        apply_rotary_emb_fn: Callable,
    ):
        """4-stream pre-alltoall pipeline. `@torch.compiler.disable(recursive=False)`:
        outer block compile region treats this as opaque (boxify) — body runs
        eager Python so `with torch.cuda.stream()` propagates to C++
        at::cuda::CUDAStream (PyTorch issue #92804 workaround). recursive=False
        allows nested torch.compile regions (the _v/q/k_pre_compiled fns) to
        re-enter inductor — preserving fusion + autotune for leaf GEMMs.

        Streams (issue order Q -> V -> K; heaviest compute on full SMs first,
        lightest gc_comp work issued early to overlap comm with K compute):
            default      : Q GEMM + qk_norm + rope (heaviest, full 148 SMs)
            gc_comp      : V GEMM (light, finishes first), then K GEMM + norm + rope
            pri_comm     : Q/V/K alltoall peer copies + merged barrier
            gc_selfcopy  : Q/V alltoall self copies (K self via op-internal
                           opSelfStream — K is last so its op carries the
                           merged 2stream_final barrier)

        Returns 5D `[P, B, Sp, H/P, D]` slot views for V/Q/K, AFTER the merged
        Q+K+V signal/wait barrier completes (default stream waits ev_done).
        Caller MUST do `post_permute_5d_to_4d` (with .contiguous()) in outer
        compile region to copy slot data out to fresh buffer for SDPA — slot
        is reused on next iter."""
        P = self.world_size
        pri_comm = self.pri_comm_stream
        gc_comp = self.gc_comp_stream
        gc_self_handle = self.gc_selfcopy_handle
        pg_boxed = self._pg_boxed
        group_ranks = self._group_ranks

        B = x.shape[0]
        Sp = x.shape[1]

        ev_v = torch.cuda.Event()
        ev_q = torch.cuda.Event()
        ev_k = torch.cuda.Event()
        ev_done = torch.cuda.Event()

        # [BISECT #1] entry-drain reverted — testing whether 3x SymMem barrier
        # V on default (full SMs, lightest: GEMM + 5D permute + contig fused).
        # Issued first so V's alltoall is in flight on pri_comm while Q (heavier)
        # computes on gc_comp — overlapping V's CE-bound peer copies with Q's
        # SM-bound RMS+RoPE compute. Pipeline order: V → Q → K.
        v_perm = self._v_pre_compiled(to_v, x, B, Sp, P, num_kv_heads, head_dim)
        ev_v.record()

        with torch.cuda.stream(pri_comm):
            ev_v.wait()
            v_5d = self._alltoall_v11_issue(v_perm, gc_self_handle)
            self._emit_barrier()

        # Q on gc_comp (~136 SMs): GEMM + norm + rope + 5D permute + contig fused
        with torch.cuda.stream(gc_comp):
            ev_v.wait()
            q_perm = self._q_pre_compiled(
                to_q, norm_q, apply_rotary_emb_fn,
                x, B, Sp, P, num_heads, head_dim, pe, rope_type)
            ev_q.record()

        with torch.cuda.stream(pri_comm):
            ev_q.wait()
            q_5d = self._alltoall_v11_issue(q_perm, gc_self_handle)
            self._emit_barrier()

        # K on gc_comp (last, carries barrier): GEMM + norm + rope + 5D permute + contig fused
        with torch.cuda.stream(gc_comp):
            k_pe_use = k_pe if k_pe is not None else pe
            k_perm = self._k_pre_compiled(
                to_k, norm_k, apply_rotary_emb_fn,
                x, B, Sp, P, num_kv_heads, head_dim, k_pe_use, rope_type)
            ev_k.record()

        with torch.cuda.stream(pri_comm):
            ev_k.wait()
            k_5d = self._alltoall_v11_issue(k_perm, gc_self_handle)
            self._emit_barrier()
            ev_done.record()

        torch.cuda.current_stream().wait_event(ev_done)
        return q_5d, k_5d, v_5d

    def forward_with_pipeline(
        self,
        x: torch.Tensor,
        to_q: torch.nn.Module,
        to_k: torch.nn.Module,
        to_v: torch.nn.Module,
        norm_q: Optional[torch.nn.Module],
        norm_k: Optional[torch.nn.Module],
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        pe: Optional[tuple],
        k_pe: Optional[tuple],
        rope_type,
        apply_rotary_emb_fn: Callable,
        **attn_kwargs,
    ) -> torch.Tensor:
        """Thin outer-compile-eligible wrapper around the @disable'd pre-attn
        pipeline. After `_pre_attn_alltoall_pipeline` returns, post_permute +
        transpose + contiguous + SDPA + output_a2a run in OUTER compile region
        (block forward re-traces here) — these get inductor fusion + autotune."""
        P = self.world_size
        q_5d, k_5d, v_5d = self._pre_attn_alltoall_pipeline(
            x, to_q, to_k, to_v, norm_q, norm_k,
            num_heads, num_kv_heads, head_dim,
            pe, k_pe, rope_type, apply_rotary_emb_fn,
        )
        v_out = post_permute_5d_to_4d(v_5d, P)
        q_out = post_permute_5d_to_4d(q_5d, P)
        k_out = post_permute_5d_to_4d(k_5d, P)

        B = q_out.shape[0]
        seq_len_full = q_out.shape[1]
        if self.inner_backend.preferred_layout == AttentionTensorLayout.HND:
            q_out = q_out.transpose(1, 2).contiguous()
            k_out = k_out.transpose(1, 2).contiguous()
            v_out = v_out.transpose(1, 2).contiguous()

        # Caller passed pre-A2A (sharded) seq_lens; the inner backend reshapes
        # by them, so hand it the post-A2A length instead (matches _forward_fused).
        attn_kwargs["seq_len"] = seq_len_full
        attn_kwargs["seq_len_kv"] = seq_len_full

        sdpa_out = self.inner_backend.forward(
            q=q_out, k=k_out, v=v_out, **attn_kwargs)
        return self._output_a2a(sdpa_out, B, seq_len_full)

    def _output_a2a(
        self,
        output: torch.Tensor,
        batch_size: int,
        seq_len_full: int,
    ) -> torch.Tensor:
        """Reverse all-to-all: [B, S, H/P, D] → [B, S/P, H, D]"""
        inner_layout = self.inner_backend.preferred_layout

        if inner_layout == AttentionTensorLayout.HND:
            output = output.transpose(1, 2).contiguous()
        else:
            if output.dim() == 3:
                output = output.view(
                    batch_size, seq_len_full, self.sharded_num_heads, self.head_dim
                )
            output = output.contiguous()

        output = all_to_all_4d(
            output, scatter_dim=1, gather_dim=2, process_group=self.process_group
        )

        return output

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Preferred tensor layout: [B, S, H, D]"""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return True


class Attention2DAttention(AttentionBackend):
    """
    Attention2D Context Parallelism wrapper for video-generation inference.

    Based on:
        "Attention2D: Communication Efficient Distributed Self-Attention Mechanism"
        https://arxiv.org/pdf/2503.15758

    The original paper targets LLM training with causal attention.  This is a
    simplified adaptation for video generation inference with full (non-causal)
    attention.

    Motivation vs. Ulysses and Ring attention
    -----------------------------------------
    *vs. Ulysses*: Ulysses (head-sharding) requires the parallelism degree to divide
    the model's head count (e.g. for WAN: degree ≤ 12 and must divide 12).
    Attention2D removes this constraint entirely — any ``row_size × col_size`` mesh
    is valid regardless of head count.  Attention2D can also be composed with Ulysses
    (head-sharding across a separate process group) to combine both parallelism axes.

    *vs. Ring attention*: Ring attention is the closest algorithmic alternative — both
    distribute sequence across GPUs without a head-count constraint.  Attention2D
    scales better: for a symmetric mesh (``row_size ≈ col_size ≈ √P``), communication
    volume scales as ``O(N / √P)`` where ``N`` is the sequence length and ``P`` is the
    total number of GPUs, compared to ``O(N)`` for ring attention.

    Mesh layout
    -----------
    Ranks are arranged in a 2-D logical mesh of shape ``[row_size, col_size]``
    (total parallelism degree = ``P = row_size * col_size``).  Each rank holds a
    ``[B, S/P, H, D]`` shard of Q, K, and V.

    Example for ``row_size=2, col_size=3`` (6 ranks total)::

                   col group (K/V all-gather)
                     ↓        ↓        ↓
                   col 0    col 1    col 2
        row 0  [  rank 0 | rank 1 | rank 2  ]  ← row group (Q all-gather)
        row 1  [  rank 3 | rank 4 | rank 5  ]  ← row group (Q all-gather)

    Ranks in the same **row** share a ``row_process_group`` and all-gather Q.
    Ranks in the same **column** share a ``col_process_group`` and all-gather K/V.

    Architecture:
        Input:   [B, S/P, H, D]  (sequence sharded across P = row_size × col_size ranks)
        Step 1:  Q all-gather within row group:        [B, S/P, H, D] → [B, S/col_size, H, D]
        Step 2:  K/V fused all-gather within col group [B, S/P, H, D] → [B, S/row_size, H, D]
                   (K and V packed into [2, B, S/P, H, D] before the gather,
                    halving NCCL launch overhead vs. two separate collectives)
        Step 3:  Local attention with inner backend:
                   Q [B, S/col_size, H, D] × K,V [B, S/row_size, H, D]
                   → output [B, S/col_size, H, D] + LSE [B, H, S/col_size]
        Step 4:  Reduce-scatter output within row group, split into:
                   all_to_all_single to exchange partial outputs and LSEs, then
                   LSE-weighted combine via flash_attn_combine
                   → [B, S/P, H, D]  (fully reduced, matching input layout)
        Output:  [B, S/P, H, D]

    Supported inner backends
    ------------------------
    The inner backend must support LSE output (``support_lse() -> True``) — required
    for the reduce-scatter combine step.  Currently the FA4 and CUTEDSL
    backends meet this requirement.

    Note: ``AttentionTensorLayout.NHD`` and ``AttentionTensorLayout.HND`` are both
    handled transparently; transposition is applied before the inner forward and
    reversed afterward.

    Note: ``support_fused_qkv()`` is *not* required — fused QKV would not reduce
    communication costs because Q and K/V are gathered over different process
    groups and cannot be merged into a single collective.

    Constraints
    -----------
    * Only ``PredefinedAttentionMask.FULL`` (or ``None``) is supported.
    * ``flash_attn_combine`` (JIT CUDA kernel) must be importable at
      construction time; the constructor raises ``ImportError`` otherwise.
    * The ``_combine`` step is wrapped in ``@torch.compiler.disable`` because
      the JIT kernel accesses raw data pointers that are incompatible with
      ``FakeTensor`` tracing during ``torch.compile``.
    """

    def __init__(
        self,
        inner_backend: AttentionBackend,
        row_process_group: torch.distributed.ProcessGroup,
        col_process_group: torch.distributed.ProcessGroup,
    ):
        self.inner_backend = inner_backend
        self.row_process_group = row_process_group
        self.col_process_group = col_process_group

        self.row_group_size = torch.distributed.get_world_size(group=row_process_group)
        self.col_group_size = torch.distributed.get_world_size(group=col_process_group)
        # Always NHD: all-gather kernels operate on [B, S/P, H, D]. Any HND conversion
        # needed by the inner backend is handled internally in forward.
        self._preferred_layout = AttentionTensorLayout.NHD

        if _flash_attn_combine is None:
            raise ImportError(
                "flash_attn_combine is not available. Attention2DAttention requires "
                "the Flash Attention JIT kernels to be built. "
                f"Import error: {_flash_attn_combine_import_error}"
            ) from _flash_attn_combine_import_error

        if not inner_backend.support_lse():
            raise RuntimeError(
                f"{type(inner_backend).__name__} does not support LSE output "
                "(support_lse() returned False). Attention2DAttention requires "
                "the inner backend to support LSE."
            )

        for attr in ("head_dim", "num_heads"):
            if not hasattr(inner_backend, attr):
                raise RuntimeError(
                    f"{type(inner_backend).__name__} is missing required attribute '{attr}'. "
                    "Attention2DAttention requires the inner backend to expose 'head_dim' and "
                    "'num_heads' as instance attributes."
                )
        self.head_dim = inner_backend.head_dim
        self.num_heads = inner_backend.num_heads
        self._inner_layout = inner_backend.preferred_layout
        if self._inner_layout not in (AttentionTensorLayout.NHD, AttentionTensorLayout.HND):
            raise NotImplementedError(
                f"{type(inner_backend).__name__} uses unsupported layout: {self._inner_layout}"
            )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with Attention2D sequence parallelism.

        q/k/v: [B, S/P, H, D] each.
        """
        B, shard_seq, H, D = q.shape
        attention_mask = kwargs.get("attention_mask", None)

        if attention_mask is not None and attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError(
                f"Attention2DAttention only supports FULL attention mask, got {attention_mask}."
            )

        if self.row_group_size > 1:
            # All-gather q within row_process_group using a single flat buffer.
            # [B, S/P, H, D] → [row_group_size, B, S/P, H, D] → [B, S/col_group_size, H, D]
            q_recv = q.new_empty(self.row_group_size, B, shard_seq, H, D)
            torch.distributed.all_gather_into_tensor(
                q_recv.view(-1), q.contiguous().view(-1), group=self.row_process_group
            )
            q = q_recv.permute(1, 0, 2, 3, 4).reshape(B, self.row_group_size * shard_seq, H, D)

        if self.col_group_size > 1:
            # Fuse K and V into a single all-gather to reduce NCCL launch overhead.
            # [2, B, S/P, H, D] → [col_group_size, 2, B, S/P, H, D] → split back to K, V
            kv_send = k.new_empty(2, B, shard_seq, H, D)
            kv_send[0].copy_(k)
            kv_send[1].copy_(v)
            kv_recv = k.new_empty(self.col_group_size, 2, B, shard_seq, H, D)
            torch.distributed.all_gather_into_tensor(
                kv_recv.view(-1), kv_send.view(-1), group=self.col_process_group
            )
            k = (
                kv_recv[:, 0]
                .permute(1, 0, 2, 3, 4)
                .reshape(B, self.col_group_size * shard_seq, H, D)
            )
            v = (
                kv_recv[:, 1]
                .permute(1, 0, 2, 3, 4)
                .reshape(B, self.col_group_size * shard_seq, H, D)
            )

        seq_len = q.shape[1]

        if self._inner_layout == AttentionTensorLayout.HND:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        output, lse = self.inner_backend.forward_with_lse(q=q, k=k, v=v, **kwargs)

        if self._inner_layout == AttentionTensorLayout.HND:
            output = output.transpose(1, 2).contiguous()
        else:
            if output.dim() == 3:
                output = output.view(B, seq_len, self.num_heads, self.head_dim)
            output = output.contiguous()

        if self.row_group_size > 1:
            # Reduce-scatter output+lse along sequence within row_process_group,
            # implemented as all_to_all_single + local LSE-based reduce.
            N = self.row_group_size
            B, seq_row, H, D = output.shape  # seq_row = S/col_group_size = N * (S/P)
            shard_seq = seq_row // N

            # output: [B, seq_row, H, D] → [N, B, shard_seq, H, D] (grouped by dest rank)
            o_send = output.view(B, N, shard_seq, H, D).permute(1, 0, 2, 3, 4).contiguous()
            o_recv = torch.empty_like(o_send)
            torch.distributed.all_to_all_single(o_recv, o_send, group=self.row_process_group)
            # o_recv: [N, B, shard_seq, H, D] — already stacked by source rank

            # lse: [B, H, seq_row] → [N, B, H, shard_seq] (grouped by dest rank)
            lse_send = lse.view(B, H, N, shard_seq).permute(2, 0, 1, 3).contiguous()
            lse_recv = torch.empty_like(lse_send)
            torch.distributed.all_to_all_single(lse_recv, lse_send, group=self.row_process_group)
            # lse_recv: [N, B, H, shard_seq] — already stacked by source rank

            # flash_attn_combine expects lse as [N, B, S/P, H] with stride(-2)==1;
            # do not call .contiguous() after permute as it would reset the strides.
            lse_recv = lse_recv.permute(0, 1, 3, 2)  # [N, B, shard_seq, H]
            output, _ = self._combine(o_recv, lse_recv, output.dtype)

        return output

    @torch.compiler.disable
    def _combine(
        self,
        o_partial: torch.Tensor,
        lse_partial: torch.Tensor,
        out_dtype: torch.dtype,
    ):
        """Combine partial attention outputs via LSE reduction.

        Isolated under @torch.compiler.disable because _flash_attn_combine is a
        JIT CUDA kernel that accesses raw data pointers, incompatible with
        FakeTensor tracing during torch.compile.
        """
        return _flash_attn_combine(
            o_partial.float().contiguous(), lse_partial, out_dtype=out_dtype, return_lse=False
        )

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        # FlashAttn4 (the only backend currently supporting the required LSE output)
        # does not support fused QKV. Even if it did, fused QKV would not reduce
        # communication costs in Attention2D since Q and K/V are gathered over
        # different process groups and cannot be fused into a single collective.
        # If a future backend supports both LSE and fused QKV with a faster kernel,
        # add fused QKV support.
        return False

    @classmethod
    def support_lse(cls) -> bool:
        return False


class RingAttention(AttentionBackend):
    """Ring sequence parallelism around an LSE-capable attention backend."""

    def __init__(
        self,
        inner_backend: AttentionBackend,
        process_group: dist.ProcessGroup,
    ):
        # Invariant: only instantiated when ring_size > 1 (see attention.py),
        # so distributed must be initialized and the group must be non-trivial.
        if not type(inner_backend).support_lse():
            raise ValueError(
                f"RingAttention requires an LSE-capable inner backend (FA4); "
                f"got {type(inner_backend).__name__}"
            )

        # Required attributes for buffer allocation in _ensure_buffers.
        for attr in ("head_dim", "num_heads"):
            if not hasattr(inner_backend, attr):
                raise RuntimeError(
                    f"{type(inner_backend).__name__} is missing required attribute "
                    f"'{attr}'. RingAttention needs the inner backend to expose "
                    "'head_dim' and 'num_heads' as instance attributes."
                )

        # Ring's _ensure_buffers / _update_out_and_lse assume NHD ([B, S, H, D]).
        # No transpose is applied around the inner forward, so an HND backend would
        # silently produce wrong results.
        if inner_backend.preferred_layout != AttentionTensorLayout.NHD:
            raise NotImplementedError(
                f"RingAttention requires an NHD inner backend; "
                f"{type(inner_backend).__name__} prefers {inner_backend.preferred_layout}."
            )

        self.inner = inner_backend
        self.pg = process_group
        self.world_size = dist.get_world_size(group=process_group)
        self.num_heads = inner_backend.num_heads
        self.num_kv_heads = getattr(inner_backend, "num_kv_heads", self.num_heads)
        self.head_dim = inner_backend.head_dim
        self._preferred_layout = AttentionTensorLayout.NHD

        # P2P ring topology cached at construction time (avoids per-step lookups).
        ring_rank = dist.get_rank(group=process_group)
        self._send_rank = dist.get_global_rank(process_group, (ring_rank + 1) % self.world_size)
        self._recv_rank = dist.get_global_rank(process_group, (ring_rank - 1) % self.world_size)
        self._send_first = ring_rank % 2 == 0
        self._p2p_reqs: list = []

        self._buf_key = None
        self._kv_bufs = None
        self._out_buf = None
        self._lse_buf = None

    def _ring_send_recv(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """Post a non-blocking neighbor exchange. Even ranks send-then-recv to
        avoid deadlock against odd ranks doing recv-then-send."""
        if self._send_first:
            ops = [
                dist.P2POp(dist.isend, send, self._send_rank, group=self.pg),
                dist.P2POp(dist.irecv, recv, self._recv_rank, group=self.pg),
            ]
        else:
            ops = [
                dist.P2POp(dist.irecv, recv, self._recv_rank, group=self.pg),
                dist.P2POp(dist.isend, send, self._send_rank, group=self.pg),
            ]
        self._p2p_reqs = dist.batch_isend_irecv(ops)

    def _ring_wait(self) -> None:
        for r in self._p2p_reqs:
            r.wait()
        self._p2p_reqs.clear()

    def _ensure_buffers(self, q: torch.Tensor, k: torch.Tensor) -> None:
        B, S, H, D = q.shape
        H_kv = k.shape[2]
        key = (B, S, H, H_kv, D, q.device, q.dtype, k.dtype)
        if key == self._buf_key:
            return
        self._kv_bufs = k.new_empty(2, 2, B, S, H_kv, D)
        # Accumulate ring blocks in fp32 to avoid repeated bf16<->fp32 rounding
        # across online-softmax merges
        self._out_buf = q.new_empty(B, S, H, D, dtype=torch.float32)
        self._lse_buf = q.new_empty(B, S, H, dtype=torch.float32)
        self._buf_key = key

    def _update_out_and_lse(
        self,
        out: torch.Tensor,
        lse: torch.Tensor,
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
    ) -> None:
        """Online-softmax merge of (out, lse) with (block_out, block_lse). In-place on out/lse."""
        c = torch.sigmoid(block_lse.unsqueeze(-1) - lse.unsqueeze(-1))
        out.sub_(c * (out - block_out))
        lse.sub_(F.logsigmoid(lse - block_lse))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.FULL,
        **kwargs,
    ) -> torch.Tensor:
        # Bypass ring for cross-attention (Q/KV seq lengths differ).
        if k.shape[1] != q.shape[1]:
            return self.inner.forward(q=q, k=k, v=v, attention_mask=attention_mask, **kwargs)
        if attention_mask != PredefinedAttentionMask.FULL:
            raise NotImplementedError(
                f"RingAttention only supports FULL attention mask, got {attention_mask}."
            )

        inner_kw = {kk: vv for kk, vv in kwargs.items() if kk != "attention_mask"}

        self._ensure_buffers(q, k)
        kv_bufs = self._kv_bufs
        out = self._out_buf
        lse = self._lse_buf

        kv_bufs[0, 0].copy_(k)
        kv_bufs[0, 1].copy_(v)
        for step in range(self.world_size):
            cur, nxt = step % 2, 1 - step % 2
            if step < self.world_size - 1:
                self._ring_send_recv(kv_bufs[cur], kv_bufs[nxt])
            block_out, block_lse_bh = self.inner.forward_with_lse(
                q=q,
                k=kv_bufs[cur, 0],
                v=kv_bufs[cur, 1],
                attention_mask=PredefinedAttentionMask.FULL,
                **inner_kw,
            )
            # Inner backend returns LSE as [B, H, S]; merge uses [B, S, H] with out [B, S, H, D].
            block_lse = block_lse_bh.transpose(1, 2).contiguous()
            if step == 0:
                out.copy_(block_out)
                lse.copy_(block_lse)
            else:
                self._update_out_and_lse(out, lse, block_out, block_lse)
            if step < self.world_size - 1:
                self._ring_wait()
        if out.dtype != q.dtype:
            return out.to(dtype=q.dtype)
        return out

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        return False

    @classmethod
    def support_lse(cls) -> bool:
        return False


def wrap_parallel_attention(
    attn: AttentionBackend,
    *,
    visual_gen_mapping: Optional["VisualGenMapping"] = None,
    enable_sequence_parallel: bool = True,
) -> AttentionBackend:
    """Wrap a compute backend with the configured parallelism strategy.

    Nesting order (inner → outer):
    - Attention2D + Ulysses: Attention2DAttention → UlyssesAttention
    - Ring + Ulysses: RingAttention → UlyssesAttention

    When ``enable_sequence_parallel`` is False, no wrappers are applied (callers
    use this for cross-attention paths that cannot use Ulysses/Ring/Attention2D).
    """
    if not enable_sequence_parallel or visual_gen_mapping is None:
        return attn

    vgm = visual_gen_mapping
    ring_size = vgm.ring_size
    ulysses_size = vgm.ulysses_size
    attn2d_size = vgm.attn2d_row_size * vgm.attn2d_col_size

    if attn2d_size > 1:
        attn = Attention2DAttention(
            inner_backend=attn,
            row_process_group=vgm.attn2d_row_group,
            col_process_group=vgm.attn2d_col_group,
        )
    elif ring_size > 1:
        attn = RingAttention(attn, process_group=vgm.ring_group)

    if ulysses_size > 1:
        attn = UlyssesAttention(attn, process_group=vgm.ulysses_group)
    return attn
