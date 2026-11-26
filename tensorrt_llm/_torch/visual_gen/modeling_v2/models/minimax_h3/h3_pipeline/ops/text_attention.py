# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class TextCausalAttention(OpWrapper):
    """Causal GQA for one packed BF16 prompt, with contiguous Q, K and V."""

    ARCHS = frozenset({Arch.SM_100})
    CELLS = (
        Cell(why="TP2 Qwen3-VL head shard", spec=dict(tokens=73, heads=32, kv_heads=4, dim=128)),
    )

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        from flash_attn.flash_attn_interface import flash_attn_varlen_func

        return flash_attn_varlen_func(
            q,
            k,
            v,
            offsets,
            offsets,
            q.shape[0],
            q.shape[0],
            dropout_p=0.0,
            softmax_scale=1 / math.sqrt(q.shape[-1]),
            causal=True,
            deterministic=False,
            return_attn_probs=False,
        )

    def reference(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        q, k, v = (tensor.float().transpose(0, 1).unsqueeze(0) for tensor in (q, k, v))
        result = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return result.squeeze(0).transpose(0, 1)

    def is_valid(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, offsets: torch.Tensor
    ) -> None:
        assert q.ndim == k.ndim == v.ndim == 3
        assert k.shape == v.shape and q.shape[0] == k.shape[0] and q.shape[2] == k.shape[2]
        assert q.shape[1] % k.shape[1] == 0
        assert q.dtype == k.dtype == v.dtype == torch.bfloat16
        assert offsets.dtype == torch.int32 and offsets.shape == (2,)
        assert offsets.device == q.device == k.device == v.device
        assert offsets.tolist() == [0, q.shape[0]]


text_causal_attention = TextCausalAttention()
