# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributed as dist

from tensorrt_llm._torch.visual_gen.runtime import AttentionTensorLayout
from tensorrt_llm._torch.visual_gen.runtime.collectives import AsyncAllToAll

from ..ops.ulysses_qkv import exchange_shards, ulysses_qkv_all_to_all, ulysses_unscatter_qkv


class H3UlyssesAttention:
    """Convert rank-local sequence rows to rank-local heads for one H3 attention call."""

    preferred_layout = AttentionTensorLayout.NHD

    def __init__(self, inner, group, stream):
        self.inner = inner
        self.group = group
        self.size = dist.get_world_size(group)
        self.hnd = inner.preferred_layout == AttentionTensorLayout.HND
        if stream is not None:
            self.transfer = AsyncAllToAll(group, stream)

    def _output(self, out, batch, seq):
        if self.hnd:
            out = out.transpose(1, 2)
        out = out.reshape(batch, seq, self.inner.num_heads, self.inner.head_dim).contiguous()
        return exchange_shards(out, self.group, scatter_dim=1, gather_dim=2)

    def _run(self, q, k, v, batch, seq, kwargs):
        kwargs.update(batch_size=batch, seq_len=seq, seq_len_kv=seq)
        return self._output(self.inner.forward(q=q, k=k, v=v, **kwargs), batch, seq)

    def forward(self, q, k, v, **kwargs):
        batch, local_rows = q.shape[:2]
        seq = local_rows * self.size
        if self.inner.support_fused_qkv():
            qkv = exchange_shards(
                torch.stack((q, k, v), dim=2), self.group, scatter_dim=3, gather_dim=1
            )
            return self._run(qkv, None, None, batch, seq, kwargs)
        q, k, v = ulysses_qkv_all_to_all(q, k, v, self.group, self.hnd)
        return self._run(q, k, v, batch, seq, kwargs)

    def forward_async(self, compute_q, compute_k, compute_v, **kwargs):
        v = self.transfer.push(compute_v())
        q = self.transfer.push(compute_q())
        k = self.transfer.push(compute_k())
        self.transfer.wait(3)
        _, batch, rows, _, _ = q.shape
        # Async prepare returns each receive at the base of its symmetric-memory allocation.
        q, k, v = ulysses_unscatter_qkv(q, k, v, self.hnd, aligned_receives=True)
        return self._run(q, k, v, batch, self.size * rows, kwargs)


def wrap_parallel_attention(inner, *, visual_gen_mapping, async_ulysses, config):
    if visual_gen_mapping is None or visual_gen_mapping.ulysses_size == 1:
        return inner
    stream = None
    if async_ulysses:
        if config.attention_metadata_state is None:
            config.attention_metadata_state = {}
        streams = config.attention_metadata_state.setdefault("h3_ulysses_streams", {})
        device = torch.cuda.current_device()
        if device not in streams:
            streams[device] = torch.cuda.Stream(device=device)
        stream = streams[device]
    return H3UlyssesAttention(inner, visual_gen_mapping.ulysses_group, stream)
