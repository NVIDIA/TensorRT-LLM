# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BF16 text-only conditioning for MiniMax-H3-Image."""

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.distributed import AllReduceParams

from ..minimax_h3.text_encoder_tp import MiniMaxH3TensorParallelTextEncoder


def _normalize(x, norm):
    # The checkpoint's learned scale applies to the rounded normalized vector,
    # not to the fp32 product. The difference accumulates over 50 layers.
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.variance_epsilon)
    return (xf * inv).to(x.dtype) * norm.weight


def _rotate(x, cos, sin):
    left, right = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-right, left), dim=-1) * sin


def _project(x, packed, widths):
    # These are views of already rank-local weights. Keep the individual GEMM
    # shapes: one larger GEMM can select a different BF16 reduction order.
    weights = packed.weight.split(widths, dim=0)
    biases = (None,) * len(weights) if packed.bias is None else packed.bias.split(widths, dim=0)
    return tuple(F.linear(x, w, b) for w, b in zip(weights, biases))


class MiniMaxH3ImageTextEncoder(MiniMaxH3TensorParallelTextEncoder):
    """Reuse Qwen weight loading, head sharding and row-parallel reductions.

    Image PDD consumes the residual stream, not a token classification. Keep
    the BF16 boundaries of the text-only Qwen3-VL prefix, without changing
    other users of the fused Qwen decoder. All three mRoPE positions coincide
    for text tokens, so only one table is needed for the whole prompt.
    """

    @torch.inference_mode()
    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        backbone = self.model.model
        x = backbone.embed_tokens(input_ids.reshape(-1).to(self.device, torch.long))
        first = backbone.layers[0].self_attn
        dim = first.head_dim
        theta = first.pos_embd_params.rope.theta
        indices = torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")
        inv_freq = (1.0 / (theta ** (indices / dim))).to(x.device)
        angles = torch.outer(
            torch.arange(x.size(0), dtype=torch.float32, device=x.device), inv_freq
        )
        angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        for layer in backbone.layers:
            attn = layer.self_attn
            h = _normalize(x, layer.input_layernorm)
            q, k, v = _project(h, attn.qkv_proj, (attn.q_size, attn.kv_size, attn.kv_size))
            q = _rotate(_normalize(q.unflatten(-1, (-1, dim)), attn.q_norm), cos, sin)
            k = _rotate(_normalize(k.unflatten(-1, (-1, dim)), attn.k_norm), cos, sin)
            v = v.unflatten(-1, (-1, dim))
            heads = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None],
                k.transpose(0, 1)[None],
                v.transpose(0, 1)[None],
                is_causal=True,
                enable_gqa=q.size(1) != k.size(1),
            )
            h = heads[0].transpose(0, 1).reshape(x.size(0), -1)
            x = x + attn.o_proj(
                h,
                all_reduce_params=AllReduceParams(
                    enable_allreduce=not layer.disable_attn_allreduce
                ),
            )
            h = _normalize(x, layer.post_attention_layernorm)
            packed = layer.mlp.gate_up_proj
            gate, up = _project(h, packed, packed.weight.size(0) // 2)
            h = F.silu(gate) * up
            x = x + layer.mlp.down_proj(
                h,
                all_reduce_params=AllReduceParams(enable_allreduce=not layer.disable_mlp_allreduce),
            )
        return x.unsqueeze(0)
