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
import torch.nn as nn

from tensorrt_llm._torch.distributed import all_to_all_4d, all_to_all_5d

from .interface import AttentionTensorLayout


class UlyssesAttention(nn.Module):
    """
    Ulysses Sequence Parallelism wrapper.

    Wraps any attention backend with sequence parallelism via all-to-all.
    Not a standalone backend — compose around a real backend (VANILLA/TRTLLM).

    Two modes:
    - fuse_qkv_a2a=False (default): 3 separate all-to-all for Q/K/V + 1 for output (4 collectives)
    - fuse_qkv_a2a=True: stacks Q/K/V into [B, S/P, 3, H, D], 1 fused 5D all-to-all
      + 1 for output (2 collectives total)
    """

    def __init__(
        self,
        inner_backend: nn.Module,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__()
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
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with Ulysses sequence parallelism.

        q/k/v: [B, S/P, H, D] each.
        When fuse_qkv_a2a=True: stacks Q/K/V → 1 fused 5D all-to-all (2 collectives)
        When fuse_qkv_a2a=False: 3 separate 4D all-to-all (4 collectives)
        """
        if self.inner_backend.support_fused_qkv():
            # default to fused QKV A2A if backend supports it.
            # This is more efficient than the unfused path.
            return self._forward_fused(q, k, v, batch_size, attention_mask)
        return self._forward_unfused(q, k, v, batch_size, attention_mask)

    def _forward_fused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Stack Q/K/V → [B, S/P, 3, H, D], then fused 5D all-to-all
        # 5D A2A is faster than 4D A2A with dim=0 or dim=-1 concat
        qkv = torch.stack([q, k, v], dim=2)
        if self.world_size > 1:
            # [B, S, 3, H/P, D]
            qkv = all_to_all_5d(qkv, scatter_dim=3, gather_dim=1, process_group=self.process_group)

        B, seq_len, _, Hp, D = qkv.shape

        # pass as fused QKV
        output = self.inner_backend.forward(
            q=qkv, k=None, v=None, batch_size=batch_size, seq_len=seq_len
        )

        return self._output_a2a(output, batch_size, seq_len)

    def _forward_unfused(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # [B, S/P, H, D] → 3 separate all-to-all → [B, S, H/P, D]
        if self.world_size > 1:
            q = all_to_all_4d(q, scatter_dim=2, gather_dim=1, process_group=self.process_group)
            k = all_to_all_4d(k, scatter_dim=2, gather_dim=1, process_group=self.process_group)
            v = all_to_all_4d(v, scatter_dim=2, gather_dim=1, process_group=self.process_group)

        seq_len_full = q.shape[1]
        inner_layout = self.inner_backend.preferred_layout

        if inner_layout == AttentionTensorLayout.HND:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

        inner_kwargs = dict(q=q, k=k, v=v, batch_size=batch_size, seq_len=seq_len_full)
        if attention_mask is not None:
            inner_kwargs["attention_mask"] = attention_mask
        output = self.inner_backend.forward(**inner_kwargs)

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
