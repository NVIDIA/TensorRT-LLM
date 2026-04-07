# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Ulysses Sequence Parallelism Wrapper

Wraps any attention backend with sequence parallelism via all-to-all
communication. Not a standalone backend — compose around a real backend
(VANILLA/TRTLLM).

Architecture:
    Input:  [B, S/P, H, D] (sequence sharded across P processes)
    Step 1: All-to-All → [B, S, H/P, D] (gather sequence, shard heads)
    Step 2: Compute attention with wrapped backend (VANILLA or TRTLLM)
    Step 3: All-to-All → [B, S/P, H, D] (restore sequence sharding)
    Output: [B, S/P, H, D] (sequence sharded)
"""

from typing import Optional

import torch

from tensorrt_llm._torch.distributed import all_to_all_4d, all_to_all_5d

from .interface import AttentionBackend, AttentionTensorLayout


class UlyssesAttention(AttentionBackend):
    """
    Ulysses Sequence Parallelism wrapper.

    Wraps any attention backend with sequence parallelism via all-to-all.
    Not a standalone backend -- compose around a real backend (VANILLA/TRTLLM).
    Fully transparent to backend-specific kwargs: everything in ``**kwargs``
    is forwarded to the inner backend unchanged (except ``seq_len`` which is
    overridden with the post-all-to-all value).

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

        if self.inner_backend.preferred_layout == AttentionTensorLayout.HND:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

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
