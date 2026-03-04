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

from tensorrt_llm._torch.distributed import all_to_all_4d

from .interface import AttentionTensorLayout


class UlyssesAttention(nn.Module):
    """
    Ulysses Sequence Parallelism wrapper.

    Wraps any attention backend with sequence parallelism via all-to-all.
    Not a standalone backend — compose around a real backend (VANILLA/TRTLLM).
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

        # Derive head info from inner backend
        self.head_dim = inner_backend.head_dim
        self.sharded_num_heads = inner_backend.num_heads
        self.sharded_num_kv_heads = getattr(inner_backend, "num_kv_heads", self.sharded_num_heads)

        # Get world size from process group
        try:
            self.world_size = torch.distributed.get_world_size(group=process_group)
        except (RuntimeError, ValueError):
            self.world_size = 1

        # Full (unsharded) head counts for external interface
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

        Input/Output: [B, S/P, H, D] (sequence sharded)

        Args:
            q: Query tensor [B, S/P, H, D]
            k: Key tensor [B, S/P, H, D]
            v: Value tensor [B, S/P, H, D]
            batch_size: Batch size
            attention_mask: Optional attention mask

        Returns:
            Output tensor [B, S/P, H, D] (sequence sharded)

        Note:
            seq_len is computed from tensor shape after all-to-all, not passed as parameter.
        """
        # Step 1: All-to-All to gather full sequence, shard heads
        # [B, S/P, H, D] -> [B, S, H/P, D]
        if self.world_size > 1:
            q = all_to_all_4d(q, scatter_dim=2, gather_dim=1, process_group=self.process_group)
            k = all_to_all_4d(k, scatter_dim=2, gather_dim=1, process_group=self.process_group)
            v = all_to_all_4d(v, scatter_dim=2, gather_dim=1, process_group=self.process_group)

        seq_len_full = q.shape[1]
        inner_layout = self.inner_backend.preferred_layout

        # Step 2: Call wrapped backend for attention
        # Transpose only if inner backend expects HND layout
        if inner_layout == AttentionTensorLayout.HND:
            # VANILLA expects [B, H/P, S, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        # NHD backends (TRTLLM) keep [B, S, H/P, D] as-is

        inner_kwargs = dict(
            q=q,
            k=k,
            v=v,
            batch_size=batch_size,
            seq_len=seq_len_full,
        )
        if attention_mask is not None:
            inner_kwargs["attention_mask"] = attention_mask
        output = self.inner_backend.forward(**inner_kwargs)

        # Convert output back to [B, S, H/P, D] for the reverse all-to-all
        if inner_layout == AttentionTensorLayout.HND:
            # VANILLA returns [B, H/P, S, D] -> transpose to [B, S, H/P, D]
            output = output.transpose(1, 2).contiguous()
        else:
            # TRTLLM returns [B, S, (H/P)*D] (3D) -> reshape to [B, S, H/P, D]
            if output.dim() == 3:
                output = output.view(
                    batch_size, seq_len_full, self.sharded_num_heads, self.head_dim
                )
            output = output.contiguous()

        # Step 3: All-to-All to restore sequence sharding
        # [B, S, H/P, D] -> [B, S/P, H, D]
        if self.world_size > 1:
            output = all_to_all_4d(
                output,
                scatter_dim=1,
                gather_dim=2,
                process_group=self.process_group,
            )

        return output

    @property
    def preferred_layout(self) -> AttentionTensorLayout:
        """Preferred tensor layout: [B, S, H, D]"""
        return self._preferred_layout

    @classmethod
    def support_fused_qkv(cls) -> bool:
        """This backend does not support fused QKV."""
        return False
