# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 drafter layer's attention on the TRTLLM backend: RoPE, KV-cache append and causal attention."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    PredefinedAttentionMask,
)
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType


class M3DraftAttention(OpWrapper):
    """The drafter layer's attention over its fused QKV rows ``qkv`` [T, (q + 2 kv) heads x 128] (bf16): the
    backend applies the drafter's RoPE (fused), appends K / V to the drafter layer's paged cache (FP8, the target's
    NVFP4 cache keeps its draft layer in FP8) and runs causal attention over the step's live rows (``metadata``
    ``num_tokens``); returns [num_tokens, q heads x 128] bf16.

    ``backend`` (``TrtllmAttention``) and ``metadata`` are the runtime's objects for the drafter layer and step.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="first draft pass of a 4-request verify (4 x 4 rows)", spec=dict(reqs=4, qlen=4)),
        Cell(why="a later draft pass, one row per request", spec=dict(reqs=8, qlen=1)),
        Cell(why="a context chunk's draft rows", spec=dict(reqs=1, context=512)),
    )

    note = """
    `reference` is exact (float64) causal softmax attention: Q rotated by the backend's NeoX RoPE at each token's
    position, over the drafter layer's K / V rows read back from its FP8 cache (dequantized by the backend's
    ``kv_scale_quant_orig``). What remains between the two is the FMHA's own arithmetic (Q's rounding to the
    kernel's input precision, its P rounding and accumulation order), which the test bounds.
    """

    def __call__(self, backend, qkv: torch.Tensor, metadata) -> torch.Tensor:
        num_tokens = metadata.num_tokens
        out = backend.forward(
            qkv[:num_tokens, :],
            None,
            None,
            metadata,
            forward_args=AttentionForwardArgs(attention_mask=PredefinedAttentionMask.CAUSAL),
        )
        return out[0] if isinstance(out, tuple) else out

    def reference(self, backend, qkv: torch.Tensor, metadata) -> torch.Tensor:
        """The call's [num_tokens, q heads x 128] output, in ``qkv``'s dtype.

        Query token ``j`` of request ``r`` sits at position ``cached_r + j`` (the metadata's cached lengths) and
        attends positions up to it. Its Q is rotated here; every K / V row, the step's own included, is read from
        the drafter layer's paged cache at the slots the metadata's KV cache manager holds, so this is evaluated
        after the call (whose roped, quantized append it then reads back; the test checks that append itself).
        Mirrors one KV head per rank and un-scaled NeoX RoPE, the drafter's configuration.
        """
        manager = metadata.kv_cache_manager
        layer = backend.layer_idx
        heads, dim = backend.num_heads, backend.head_dim
        assert backend.num_kv_heads == 1, "mirrors the one-KV-head-per-rank layout only"
        rope = backend.rope_params
        assert backend.position_embedding_type == int(PositionEmbeddingType.rope_gpt_neox)
        assert rope.scale_type == RotaryScalingType.none and rope.alpha == 1.0
        page = int(manager.tokens_per_block)
        sm_scale = dim**-0.5 / float(backend.q_scaling)
        kv_scale = float(backend.kv_scale_quant_orig.reshape(-1)[0])
        num_seqs = int(metadata.num_seqs)
        qlens = [int(n) for n in metadata.seq_lens[:num_seqs].tolist()]
        cached = [int(c) for c in metadata.kv_cache_params.num_cached_tokens_per_seq[:num_seqs]]
        total = int(metadata.num_tokens)
        assert sum(qlens) == total
        device = qkv.device

        rot = int(rope.dim)
        inv_freq = float(rope.theta) ** (
            -torch.arange(0, rot, 2, dtype=torch.float64, device=device) / rot
        )
        pool = manager.get_buffers(layer, "HND")
        q = qkv[:total, : heads * dim].to(torch.float64).reshape(total, heads, dim)
        out = torch.zeros(total, heads, dim, dtype=torch.float64, device=device)
        first = 0
        for row, request_id in enumerate(list(metadata.request_ids)[:num_seqs]):
            n, length = qlens[row], cached[row] + qlens[row]
            pages = torch.tensor(
                manager.get_batch_cache_indices([request_id], layer_idx=layer)[0][
                    : (length + page - 1) // page
                ],
                dtype=torch.int64,
                device=device,
            )
            keys = pool[pages, 0, 0].to(torch.float64).reshape(-1, dim)[:length] * kv_scale
            values = pool[pages, 1, 0].to(torch.float64).reshape(-1, dim)[:length] * kv_scale
            qpos = cached[row] + torch.arange(n, device=device)
            angles = qpos.to(torch.float64)[:, None] * inv_freq[None, :]
            cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
            rows = q[first : first + n]
            x1, x2 = rows[..., : rot // 2], rows[..., rot // 2 : rot]
            rows = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, rows[..., rot:]], dim=-1)
            allowed = torch.arange(length, device=device)[None, :] <= qpos[:, None]
            logits = torch.einsum("thd,kd->thk", rows, keys) * sm_scale
            logits = logits.masked_fill(~allowed[:, None, :], float("-inf"))
            out[first : first + n] = torch.softmax(logits, dim=-1) @ values
            first += n
        return out.reshape(total, heads * dim).to(qkv.dtype)


m3_draft_attention = M3DraftAttention()
