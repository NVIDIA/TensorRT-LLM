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

from typing import TYPE_CHECKING, Callable, ClassVar, Dict, Optional

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


def post_permute_5d_to_4d(out_5d, P):
    """5D [P, B, Sp, H/P, D] → 4D [B, P*Sp, H/P, D] (block-by-rank gather).
    .contiguous() copies slot data out (layout-normalize for SDPA, decoupling
    from IPC slot lifetime). Inductor fuses permute+contig with downstream
    SDPA input prep."""
    _P, Bt, Spt, HpP, Dt = out_5d.shape
    return out_5d.permute(1, 0, 2, 3, 4).contiguous().view(Bt, _P * Spt, HpP, Dt)


def _ulysses_post_unscatter(q_5d, k_5d, v_5d, *, is_hnd):
    """One-launch fused replacement for the post-A2A 5D -> 4D chain.

    is_hnd=True  -> output [B, H, P*Sp, D] (VANILLA / torch SDPA)
    is_hnd=False -> output [B, P*Sp, H, D] (TRTLLM / FA4)
    """
    layout = 0 if is_hnd else 1
    return torch.ops.trtllm.ulysses_post_unscatter_qkv(q_5d, k_5d, v_5d, layout)


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

    # One side stream shared across all UlyssesAttention instances on the
    # same device. Per-layer streams inflate the stream count and break
    # cuda_graph capture.
    _side_stream_by_device: ClassVar[Dict[int, "torch.cuda.Stream"]] = {}

    def __init__(
        self,
        inner_backend: AttentionBackend,
        process_group: torch.distributed.ProcessGroup,
        async_ulysses: bool = False,
    ):
        self.inner_backend = inner_backend
        self.process_group = process_group
        self._preferred_layout = AttentionTensorLayout.NHD

        self.head_dim = inner_backend.head_dim
        self.sharded_num_heads = inner_backend.num_heads
        self.sharded_num_kv_heads = getattr(inner_backend, "num_kv_heads", self.sharded_num_heads)

        self.world_size = torch.distributed.get_world_size(group=process_group)

        self.num_heads = self.sharded_num_heads * self.world_size
        self.num_kv_heads = self.sharded_num_kv_heads * self.world_size

        # Async pipeline state. Eagerly populated when async_ulysses=True;
        # forward_async assumes these are set. Non-async path doesn't touch
        # them.
        self._pg_boxed = None
        self._async_side_stream: Optional[torch.cuda.Stream] = None
        # Count of deferred pushes since the last `_join_async`. `_join_async`
        # drains exactly this many `ulysses_a2a_async_barrier` calls on the
        # side stream so V/Q/K pushes FIFO together without intermediate
        # barrier kernels.
        self._pending_barriers: int = 0
        if async_ulysses:
            device = torch.cuda.current_device()
            if device not in UlyssesAttention._side_stream_by_device:
                UlyssesAttention._side_stream_by_device[device] = torch.cuda.Stream(device=device)
            self._async_side_stream = UlyssesAttention._side_stream_by_device[device]
            if process_group is not None:
                self._pg_boxed = process_group.boxed()

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
        gate_compress = kwargs.pop("gate_compress", None)
        gate_fine = kwargs.pop("gate_fine", None)

        batch_size = q.shape[0]
        qkv = torch.stack([q, k, v], dim=2)
        qkv = all_to_all_5d(qkv, scatter_dim=3, gather_dim=1, process_group=self.process_group)

        B, seq_len, _, Hp, D = qkv.shape

        if gate_compress is not None:
            gate_compress = all_to_all_4d(
                gate_compress, scatter_dim=2, gather_dim=1, process_group=self.process_group
            )
        if gate_fine is not None:
            gate_fine = all_to_all_4d(
                gate_fine, scatter_dim=2, gather_dim=1, process_group=self.process_group
            )

        # Caller passed pre-A2A (sharded) seq_len; the inner backend
        # reshapes by it, so hand it the post-A2A length instead.
        kwargs["batch_size"] = batch_size
        kwargs["seq_len"] = seq_len
        kwargs["seq_len_kv"] = seq_len
        if gate_compress is not None:
            kwargs["gate_compress"] = gate_compress
        if gate_fine is not None:
            kwargs["gate_fine"] = gate_fine

        output = self.inner_backend.forward(q=qkv, k=None, v=None, **kwargs)

        return self._output_a2a(output, batch_size, seq_len)

    def _forward_unfused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # gate_compress / gate_fine (VSA) must follow the same all-to-all as
        # Q/K/V so they arrive at the inner backend in the same (full-S, sharded-H) layout.
        gate_compress = kwargs.pop("gate_compress", None)
        gate_fine = kwargs.pop("gate_fine", None)

        batch_size = q.shape[0]
        q = all_to_all_4d(q, scatter_dim=2, gather_dim=1, process_group=self.process_group)
        k = all_to_all_4d(k, scatter_dim=2, gather_dim=1, process_group=self.process_group)
        v = all_to_all_4d(v, scatter_dim=2, gather_dim=1, process_group=self.process_group)
        if gate_compress is not None:
            gate_compress = all_to_all_4d(
                gate_compress, scatter_dim=2, gather_dim=1, process_group=self.process_group
            )
        if gate_fine is not None:
            gate_fine = all_to_all_4d(
                gate_fine, scatter_dim=2, gather_dim=1, process_group=self.process_group
            )

        seq_len_full = q.shape[1]
        kv_seq_len_full = k.shape[1]

        if self.inner_backend.preferred_layout == AttentionTensorLayout.HND:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if gate_compress is not None:
                gate_compress = gate_compress.transpose(1, 2)
            if gate_fine is not None:
                gate_fine = gate_fine.transpose(1, 2)

        # Caller passed pre-A2A (sharded) seq_lens; hand the inner
        # backend the post-A2A lengths instead.
        kwargs["batch_size"] = batch_size
        kwargs["seq_len"] = seq_len_full
        kwargs["seq_len_kv"] = kv_seq_len_full
        if gate_compress is not None:
            kwargs["gate_compress"] = gate_compress
        if gate_fine is not None:
            kwargs["gate_fine"] = gate_fine

        output = self.inner_backend.forward(q=q, k=k, v=v, **kwargs)

        return self._output_a2a(output, batch_size, seq_len_full)

    # ------------------------------------------------------------------
    # Split-QKV async A2A pipeline. `_issue_async` and `_join_async` are
    # the only stream-switch boundaries and are @torch.compiler.disable'd;
    # the caller's compiled forward fuses each compute_{q,k,v} closure.
    # ------------------------------------------------------------------

    @torch.compiler.disable(recursive=False)
    def _issue_async(self, perm_4d: torch.Tensor) -> torch.Tensor:
        """Issue one V/Q/K async a2a (CE push only; barrier deferred to join).
        Phase 1 (acquire slot + CUDA C permute+scatter) runs on the CURRENT
        (default) stream. Phase 2a (cudaMemcpyBatchAsync peer push) is queued
        on the comm side stream, gated by an event so it waits for Phase 1.
        Phase 2b (symm-mem barrier) is NOT issued here — `_join_async` drains
        all pending barriers in one shot so V/Q/K pushes FIFO through CE
        without intermediate barrier kernels splitting them up. Returns the
        5D recv-buf view.

        Comm-stream FIFO serializes consecutive V/Q/K pushes in caller order;
        no explicit chain event is needed between them. The default stream
        is free to immediately begin the next V/Q/K compute — that's where
        the V_push ∥ Q_compute ∥ K_compute overlap comes from."""
        recv, send_h = torch.ops.trtllm.ulysses_a2a_async_prepare(perm_4d, self._pg_boxed)
        ev = torch.cuda.Event()
        ev.record()
        with torch.cuda.stream(self._async_side_stream):
            ev.wait()
            torch.ops.trtllm.ulysses_a2a_async_push(send_h, self._pg_boxed)
        self._pending_barriers += 1
        return recv

    @torch.compiler.disable(recursive=False)
    def _join_async(self) -> None:
        """Drain pending symm-mem barriers (one per deferred push) on the
        side stream, then have the default stream wait on the tail event.
        Comm-stream FIFO preserves [push V, push Q, push K, barrier, barrier,
        barrier] order; all N barriers fire on channel=0 with identical
        semantics, so the default stream sees a fully-synced recv buffer."""
        with torch.cuda.stream(self._async_side_stream):
            for _ in range(self._pending_barriers):
                torch.ops.trtllm.ulysses_a2a_async_barrier(self._pg_boxed)
            ev_done = torch.cuda.Event()
            ev_done.record()
        self._pending_barriers = 0
        torch.cuda.current_stream().wait_event(ev_done)

    def forward_async(
        self,
        compute_q: Callable[[], torch.Tensor],
        compute_k: Callable[[], torch.Tensor],
        compute_v: Callable[[], torch.Tensor],
        issue_order: tuple = ("v", "q", "k"),
        **attn_kwargs,
    ) -> torch.Tensor:
        """Run the async ulysses attention path (Q/K/V rolling A2A).

        Args:
            compute_q / compute_k / compute_v : caller-provided closures that
                each return a 4D tensor `[B, S_local, H, D]`. The closure
                typically does `GEMM → (RMSNorm) → (RoPE) → view(4D)`; closures
                live in the caller's compiled forward so inductor fuses each
                into a single Triton kernel.
            issue_order : order in which the three closures are computed +
                issued. Default `("v", "q", "k")` (self-attn). Cross-attn passes
                `("q", "k", "v")` to issue the small audio-Q first. Order is
                correctness-neutral (`_join_async` syncs all recv bufs).
            **attn_kwargs : forwarded to the wrapped inner attention backend
                (mask, scale, etc.).

        Returns:
            output tensor in the caller's sharded layout `[B, S/P, H, D]`.

        Pipeline: the three closures run on the default stream in `issue_order`;
        each compute's output is fed to `_issue_async` which queues push+barrier
        on the comm side stream. Default stream proceeds to the next compute
        immediately, so each push overlaps with the next compute. `_join_async`
        makes default wait on the last push.
        Post-attention permute / SDPA / reverse A2A run in the caller's outer
        compile region for additional inductor fusion."""
        P = self.world_size

        # Issue the closures in issue_order. Order is correctness-neutral (_join_async
        # syncs all recv bufs); it only tunes which push overlaps which compute.
        computes = {"q": compute_q, "k": compute_k, "v": compute_v}
        recv = {}
        for name in issue_order:
            recv[name] = self._issue_async(computes[name]())
        self._join_async()
        q_5d, k_5d, v_5d = recv["q"], recv["k"], recv["v"]

        # Fast path: one fused kernel replaces the eager post-A2A chain
        # (6 ops for HND target: permute+reshape+contig + transpose+contig
        # per Q/K/V; 3 ops for NHD target). bf16-only because the kernel is
        # only instantiated for __nv_bfloat16.
        _, B_q, Sp_q, HpP_q, D_q = q_5d.shape
        is_hnd = self.inner_backend.preferred_layout == AttentionTensorLayout.HND
        use_fused_post_unscatter = q_5d.dtype == torch.bfloat16
        if use_fused_post_unscatter:
            q_out, k_out, v_out = _ulysses_post_unscatter(q_5d, k_5d, v_5d, is_hnd=is_hnd)
            B = B_q
            seq_len_full = P * Sp_q
            seq_len_kv_full = P * k_5d.shape[2]  # cross-attn: K/V seq (Sp_k) differs from Q
        else:
            v_out = post_permute_5d_to_4d(v_5d, P)
            q_out = post_permute_5d_to_4d(q_5d, P)
            k_out = post_permute_5d_to_4d(k_5d, P)

            B = q_out.shape[0]
            seq_len_full = q_out.shape[1]
            seq_len_kv_full = k_out.shape[1]  # cross-attn: K/V seq differs from Q
            if is_hnd:
                q_out = q_out.transpose(1, 2).contiguous()
                k_out = k_out.transpose(1, 2).contiguous()
                v_out = v_out.transpose(1, 2).contiguous()

        attn_kwargs["seq_len"] = seq_len_full
        attn_kwargs["seq_len_kv"] = seq_len_kv_full
        output = self.inner_backend.forward(q=q_out, k=k_out, v=v_out, **attn_kwargs)
        return self._output_a2a(output, B, seq_len_full)

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
    ``[B, S_q/P, H_q, D]`` shard of Q and ``[B, S_kv/P, H_kv, D]`` shards of K and V.
    For self-attention ``S_q = S_kv`` and ``H_q = H_kv``; for GQA ``H_kv < H_q``; for
    cross-attention ``S_kv`` may differ from ``S_q``.  K/V must be sequence-sharded
    across the same mesh as Q (not replicated on every rank).

    Example for ``row_size=2, col_size=3`` (6 ranks total)::

                   col group (K/V all-gather)
                     ↓        ↓        ↓
                   col 0    col 1    col 2
        row 0  [  rank 0 | rank 1 | rank 2  ]  ← row group (Q all-gather)
        row 1  [  rank 3 | rank 4 | rank 5  ]  ← row group (Q all-gather)

    Ranks in the same **row** share a ``row_process_group`` and all-gather Q.
    Ranks in the same **column** share a ``col_process_group`` and all-gather K/V.

    Architecture:
        Input:   Q [B, S_q/P, H_q, D], K/V [B, S_kv/P, H_kv, D]
                 (sequence sharded across P = row_size × col_size ranks)
        Step 1:  Q all-gather within row group:
                   [B, S_q/P, H_q, D] → [B, S_q/row_size, H_q, D]
        Step 2:  K/V fused all-gather within col group:
                   [B, S_kv/P, H_kv, D] → [B, S_kv/col_size, H_kv, D]
                   (K and V packed into [2, B, S_kv/P, H_kv, D] before the gather,
                    halving NCCL launch overhead vs. two separate collectives)
        Step 3:  Local attention with inner backend:
                   Q [B, S_q/row_size, H_q, D] × K,V [B, S_kv/col_size, H_kv, D]
                   → output [B, S_q/row_size, H_q, D] + LSE [B, H_q, S_q/row_size]
        Step 4:  Reduce-scatter output within row group, split into:
                   all_to_all_single to exchange partial outputs and LSEs, then
                   LSE-weighted combine via flash_attn_combine
                   → [B, S_q/P, H_q, D]  (fully reduced, matching input Q layout)
        Output:  [B, S_q/P, H_q, D]

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
    * Global ``S_q`` and ``S_kv`` must each be divisible by ``P = row_size × col_size``
      so every rank holds an equal local shard.
    * Cross-attention requires K/V to be sequence-sharded across the mesh (same as Q),
      not replicated on every rank.
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
        self.num_kv_heads = getattr(inner_backend, "num_kv_heads", self.num_heads)
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

        q: [B, S_q/P, H_q, D].  k/v: [B, S_kv/P, H_kv, D].
        """
        B, shard_seq_q, H_q, D = q.shape
        _, shard_seq_kv, H_kv, D_kv = k.shape
        attention_mask = kwargs.get("attention_mask", None)

        if D_kv != D:
            raise ValueError(
                f"Attention2DAttention: q head_dim ({D}) must match k head_dim ({D_kv})."
            )
        if v.shape != k.shape:
            raise ValueError(
                f"Attention2DAttention: k and v shapes must match, got k={k.shape}, v={v.shape}."
            )
        if H_q != self.num_heads:
            raise ValueError(
                f"Attention2DAttention: q num_heads ({H_q}) must match "
                f"inner backend num_heads ({self.num_heads})."
            )
        if H_kv != self.num_kv_heads:
            raise ValueError(
                f"Attention2DAttention: k num_kv_heads ({H_kv}) must match "
                f"inner backend num_kv_heads ({self.num_kv_heads})."
            )

        if attention_mask is not None and attention_mask != PredefinedAttentionMask.FULL:
            raise ValueError(
                f"Attention2DAttention only supports FULL attention mask, got {attention_mask}."
            )

        if self.row_group_size > 1:
            # All-gather q within row_process_group using a single flat buffer.
            # [B, S_q/P, H_q, D] → [row_group_size, B, S_q/P, H_q, D]
            # → [B, S_q/row_size, H_q, D]
            q_recv = q.new_empty(self.row_group_size, B, shard_seq_q, H_q, D)
            torch.distributed.all_gather_into_tensor(
                q_recv.view(-1), q.contiguous().view(-1), group=self.row_process_group
            )
            q = q_recv.permute(1, 0, 2, 3, 4).reshape(B, self.row_group_size * shard_seq_q, H_q, D)

        if self.col_group_size > 1:
            # Fuse K and V into a single all-gather to reduce NCCL launch overhead.
            # [2, B, S_kv/P, H_kv, D] → [col_group_size, 2, B, S_kv/P, H_kv, D]
            # → [B, S_kv/col_size, H_kv, D]
            kv_send = k.new_empty(2, B, shard_seq_kv, H_kv, D)
            kv_send[0].copy_(k)
            kv_send[1].copy_(v)
            kv_recv = k.new_empty(self.col_group_size, 2, B, shard_seq_kv, H_kv, D)
            torch.distributed.all_gather_into_tensor(
                kv_recv.view(-1), kv_send.view(-1), group=self.col_process_group
            )
            k = (
                kv_recv[:, 0]
                .permute(1, 0, 2, 3, 4)
                .reshape(B, self.col_group_size * shard_seq_kv, H_kv, D)
            )
            v = (
                kv_recv[:, 1]
                .permute(1, 0, 2, 3, 4)
                .reshape(B, self.col_group_size * shard_seq_kv, H_kv, D)
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
    use_ulysses: bool = True,
    async_ulysses: bool = False,
) -> AttentionBackend:
    """Wrap a compute backend with the configured parallelism strategy.

    Nesting order (inner → outer):
    - Attention2D + Ulysses: Attention2DAttention → UlyssesAttention
    - Ring + Ulysses: RingAttention → UlyssesAttention

    When ``enable_sequence_parallel`` is False, no wrappers are applied (callers
    use this for cross-attention paths that cannot use Ulysses/Ring/Attention2D).

    ``use_ulysses`` gates the Ulysses head-sharding wrap independently of
    ``ulysses_size``: pass False to skip it even when ``ulysses_size > 1`` (e.g. a
    SEPARATE_QKV cross-attn that falls back to all-gather and built its inner
    backend with the full, un-sharded head count). This keeps the wrap consistent
    with the caller's inner head-count decision.
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

    if ulysses_size > 1 and use_ulysses:
        attn = UlyssesAttention(
            attn,
            process_group=vgm.ulysses_group,
            async_ulysses=async_ulysses,
        )
    return attn
