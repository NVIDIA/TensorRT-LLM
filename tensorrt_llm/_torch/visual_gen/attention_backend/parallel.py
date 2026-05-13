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
backend — compose around a real backend (VANILLA/TRTLLM/FA4).

"""

from typing import Optional

import torch

from tensorrt_llm._torch.distributed import all_to_all_4d, all_to_all_5d

from ...attention_backend.interface import PredefinedAttentionMask
from .interface import AttentionBackend, AttentionTensorLayout

_flash_attn_combine_import_error = None
try:
    from tensorrt_llm._torch.visual_gen.jit_kernels.flash_attention.cute.interface import (
        flash_attn_combine as _flash_attn_combine,
    )
except (ImportError, OSError) as e:
    _flash_attn_combine = None
    _flash_attn_combine_import_error = e


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
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        self.inner_backend = inner_backend
        self.process_group = process_group
        self._preferred_layout = AttentionTensorLayout.NHD

        self.head_dim = inner_backend.head_dim
        self.sharded_num_heads = inner_backend.num_heads
        self.sharded_num_kv_heads = getattr(inner_backend, "num_kv_heads", self.sharded_num_heads)

        try:
            self.world_size = torch.distributed.get_world_size(group=process_group)
        except (RuntimeError, ValueError):
            self.world_size = 1

        self.num_heads = self.sharded_num_heads * self.world_size
        self.num_kv_heads = self.sharded_num_kv_heads * self.world_size

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
        if self.world_size > 1:
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
        if self.world_size > 1:
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

        if self.world_size > 1:
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
    for the reduce-scatter combine step.  Currently only the FA4 backend
    (``FlashAttn4Attention``) meets this requirement.

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
