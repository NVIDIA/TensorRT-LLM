# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam decoder blocks 0-2 / sm_103 / tp1: one dense, sliding-attention block, constructed directly.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no ``ForCausalLM``: nothing in the engine builds this class, and it serves no request.
A caller constructs it from the checkpoint's configuration for one of the dense blocks
0, 1 or 2, loads the block's fifteen tensors and runs one step of one BF16 request through
it -- a context step after any cached prefix, or a one-token generation step -- with the
attention batch state prepared on a ``KVCacheManagerV2``. The forward is
assembled from catalog entries, with the attention sublayer shared with the sparse block
component in ``../_attention.py``, and it does not call the built-in Beam modeling.

The block, with block 0's residual coefficients (block i's are in ``depth_scales``)::

    h = rmsnorm(x, input_layernorm)
    q, k, v, g = h @ q_proj^T, h @ k_proj^T, h @ v_proj^T, h @ gate_proj^T
    q, k = rmsnorm(q, q_norm), rmsnorm(k, k_norm)          per head, rounded to bf16
    q, k = rope(q, k)                                       fp32 NeoX rotation, rounded
    a = causal attention(q, k, v), window 512, scale 1/sqrt(128), 6 q heads per kv head
    a = a * sigmoid(g + gate_bias)                          per head, bf16 steps
    x = rmsnorm(a @ o_proj^T, post_attention_layernorm) / sqrt(2) + x / sqrt(2)
    h = rmsnorm(x, pre_feedforward_layernorm)
    m = (silu(h @ gate^T) * (h @ up^T)) @ down^T            silu and product in fp32
    out = rmsnorm(m, post_feedforward_layernorm) / sqrt(3) + x * sqrt(2) / sqrt(3)

Every RMSNorm computes its statistics, normalization and affine in fp32 and rounds once.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.beam_depth_residual import (
    beam_depth_residual,
    depth_scales,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

from ..full_model_nvfp4__sm_103__tp4 import _attention

#: The configuration this component computes, as the checkpoint's config.json states it.
#: Anything else is rejected at construction rather than computed approximately.
SUPPORTED_CONFIG = {
    "hidden_size": 5120,
    "intermediate_size": 20480,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "sliding_window": 512,
}
SUPPORTED_LAYER_IDX = 0
#: The dense blocks of the checkpoint's schedule; ``layer_idx`` fixes the residual coefficients.
SUPPORTED_LAYER_INDICES = (0, 1, 2)
SUPPORTED_ROPE = {"rope_type": "default", "rope_theta": 10000.0}

#: The most tokens one call may append: the six-layer model's sequence capacity. Longer
#: calls are rejected; the request's whole KV length is bounded by its cache's capacity.
MAX_TOKENS = 1024

_SM = (10, 3)


def weight_shapes(config) -> "OrderedDict[str, tuple[int, ...]]":
    """The block's fifteen tensors by checkpoint name, in checkpoint order, [out, in] for matrices."""
    hidden, inter = config.hidden_size, config.intermediate_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    return OrderedDict(
        [
            ("input_layernorm.weight", (hidden,)),
            ("self_attn.gate_bias", (config.num_attention_heads,)),
            ("self_attn.q_proj.weight", (q_width, hidden)),
            ("self_attn.k_proj.weight", (kv_width, hidden)),
            ("self_attn.v_proj.weight", (kv_width, hidden)),
            ("self_attn.gate_proj.weight", (q_width, hidden)),
            ("self_attn.o_proj.weight", (hidden, q_width)),
            ("self_attn.q_norm.weight", (config.head_dim,)),
            ("self_attn.k_norm.weight", (config.head_dim,)),
            ("pre_feedforward_layernorm.weight", (hidden,)),
            ("mlp.gate_proj.weight", (inter, hidden)),
            ("mlp.up_proj.weight", (inter, hidden)),
            ("mlp.down_proj.weight", (hidden, inter)),
            ("post_attention_layernorm.weight", (hidden,)),
            ("post_feedforward_layernorm.weight", (hidden,)),
        ]
    )


def _attribute(name: str) -> str:
    return name.removesuffix(".weight").replace(".", "_")


def check_config(config, layer_idx: int) -> None:
    """Raise ValueError naming the first field outside what this component computes."""
    if type(layer_idx) is not int or layer_idx not in SUPPORTED_LAYER_INDICES:
        raise ValueError(
            f"only dense blocks {SUPPORTED_LAYER_INDICES} are supported, got layer_idx={layer_idx!r}"
        )
    for field, expected in SUPPORTED_CONFIG.items():
        actual = getattr(config, field, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"unsupported {field}={actual!r}; this block computes {expected!r}")
    layer_types = getattr(config, "layer_types", None) or []
    if len(layer_types) <= layer_idx or layer_types[layer_idx] != "sliding_attention":
        raise ValueError(
            f"block {layer_idx} must be a sliding_attention layer: "
            f"{layer_types[layer_idx] if len(layer_types) > layer_idx else None!r}"
        )
    mlp_types = getattr(config, "mlp_layer_types", None) or []
    if len(mlp_types) <= layer_idx or mlp_types[layer_idx] != "dense":
        raise ValueError(
            f"block {layer_idx} must have a dense MLP: "
            f"{mlp_types[layer_idx] if len(mlp_types) > layer_idx else None!r}"
        )
    rope = (getattr(config, "rope_parameters", None) or {}).get("sliding_attention")
    if rope != SUPPORTED_ROPE:
        raise ValueError(
            f"unsupported sliding_attention rope parameters {rope!r}; need {SUPPORTED_ROPE!r}"
        )


class BeamDenseBlockBf16Sm103Tp1(nn.Module):
    """Beam dense decoder block 0, 1 or 2: BF16 weights and activations, one GPU, one request step.

    ``forward(hidden_states [1, T, 5120] bf16, position_ids [1, T], attn_metadata)``
    returns ``[1, T, 5120]`` bf16 for 1 <= T <= ``MAX_TOKENS``, a prepared
    ``TrtllmAttentionMetadata`` holding one step of one request on a BF16
    ``KVCacheManagerV2`` (8 kv heads of 128) that holds the block's logical layer
    ``layer_idx`` -- a context step of T tokens after ``past`` cached ones (paged-context
    FMHA when ``past > 0``) or a one-token generation step -- and positions
    ``arange(past, past + T)``. Inputs outside that are rejected before any kernel runs;
    batches of several requests and other blocks are not computed here.
    """

    def __init__(
        self, config, layer_idx: int = SUPPORTED_LAYER_IDX, device: str | torch.device = "cuda"
    ):
        super().__init__()
        check_config(config, layer_idx)
        device = torch.device(device)
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != _SM:
            raise ValueError(
                f"this block is built for sm_{_SM[0]}{_SM[1]} CUDA devices, got {device}"
            )
        self.layer_idx = layer_idx
        self.eps = config.rms_norm_eps
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.window = config.sliding_window
        self.rope_theta = SUPPORTED_ROPE["rope_theta"]
        self.attention_scales = depth_scales(layer_idx, is_attention=True)
        self.feed_forward_scales = depth_scales(layer_idx, is_attention=False)
        self._shapes = weight_shapes(config)
        for name, shape in self._shapes.items():
            tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self.register_parameter(_attribute(name), nn.Parameter(tensor, requires_grad=False))
        self._loaded = False

    def checkpoint_tensors(self) -> "OrderedDict[str, torch.Tensor]":
        """The block's tensors by checkpoint name, in checkpoint order."""
        return OrderedDict((name, getattr(self, _attribute(name))) for name in self._shapes)

    @torch.no_grad()
    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Copy exactly the block's fifteen tensors, by checkpoint name, BF16, [out, in]."""
        if not isinstance(weights, Mapping):
            raise TypeError("weights must be a mapping")
        missing = [n for n in self._shapes if n not in weights]
        unexpected = sorted(n for n in weights if n not in self._shapes)
        if missing or unexpected:
            raise ValueError(f"block weights mismatch: missing {missing}, unexpected {unexpected}")
        for name, shape in self._shapes.items():
            tensor = weights[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"{name}: got {tuple(tensor.shape)} {tensor.dtype}, need {shape} torch.bfloat16"
                )
            if tensor.layout != torch.strided or tensor.device.type not in ("cpu", "cuda"):
                raise ValueError(f"{name} must be a dense host or CUDA tensor")
        self._loaded = False
        for name in self._shapes:
            getattr(self, _attribute(name)).copy_(weights[name])
        self._loaded = True

    def _check_step(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, md: TrtllmAttentionMetadata
    ) -> None:
        """Raise ValueError on anything this block would otherwise compute wrongly."""
        _attention.check_step(
            hidden_states,
            position_ids,
            md,
            layer_idx=self.layer_idx,
            hidden_size=self.hidden_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=self.input_layernorm.device,
            max_tokens=MAX_TOKENS,
        )
        if not self._loaded:
            raise ValueError("load_weights must complete before running the block")

    @property
    def hidden_size(self) -> int:
        return self._shapes["input_layernorm.weight"][0]

    def step_args(self, md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state."""
        return _attention.step_args(md, self.layer_idx)

    def attention(self, h: torch.Tensor, positions: torch.Tensor, step: dict) -> torch.Tensor:
        """Normalized input [T, hidden] -> attention output [T, hidden], before post_attention_layernorm."""
        return _attention.attention(
            h,
            positions,
            step,
            q_proj=self.self_attn_q_proj,
            k_proj=self.self_attn_k_proj,
            v_proj=self.self_attn_v_proj,
            gate_proj=self.self_attn_gate_proj,
            o_proj=self.self_attn_o_proj,
            q_norm=self.self_attn_q_norm,
            k_norm=self.self_attn_k_norm,
            gate_bias=self.self_attn_gate_bias,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            eps=self.eps,
            rope_theta=self.rope_theta,
            window=self.window,
        )

    def feed_forward(self, h: torch.Tensor) -> torch.Tensor:
        """Normalized input [T, hidden] -> dense SwiGLU output [T, hidden], before its post-norm."""
        gate = cublas_mm(h, self.mlp_gate_proj.t())
        up = cublas_mm(h, self.mlp_up_proj.t())
        act = flashinfer_silu_and_mul(torch.cat((gate, up), dim=-1))
        return cublas_mm(act, self.mlp_down_proj.t())

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: TrtllmAttentionMetadata,
    ) -> torch.Tensor:
        self._check_step(hidden_states, position_ids, attn_metadata)
        step = self.step_args(attn_metadata)
        num_tokens = hidden_states.shape[1]
        x = hidden_states.view(num_tokens, self.hidden_size)
        positions = position_ids.view(num_tokens)

        h = flashinfer_rmsnorm(x, self.input_layernorm, self.eps)
        h = self.attention(h, positions, step)
        h = flashinfer_rmsnorm(h, self.post_attention_layernorm, self.eps)
        x = beam_depth_residual(h, x, *self.attention_scales)

        h = flashinfer_rmsnorm(x, self.pre_feedforward_layernorm, self.eps)
        h = self.feed_forward(h)
        h = flashinfer_rmsnorm(h, self.post_feedforward_layernorm, self.eps)
        x = beam_depth_residual(h, x, *self.feed_forward_scales)
        return x.view(1, num_tokens, self.hidden_size)
