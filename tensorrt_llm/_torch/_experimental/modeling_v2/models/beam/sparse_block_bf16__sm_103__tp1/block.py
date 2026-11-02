# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam decoder blocks 3-5 / sm_103 / tp1: one sparse (MoE) block, constructed directly.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no ``ForCausalLM``: nothing in the engine builds this class, and it serves no request.
A caller constructs it from the checkpoint's configuration for one of the sparse blocks
3, 4 or 5, loads the block's 782 tensors and runs one step of one BF16 request through it
-- a context step after any cached prefix, or a one-token generation step -- with the
attention batch state prepared on a ``KVCacheManagerV2`` that holds the block's logical
layer. The forward is assembled from catalog entries, the attention sublayer
shared with the dense block component (``../_attention.py``) and the MoE component
(``../moe_bf16__sm_103__tp1``); it does not call the built-in Beam modeling and owns no
dense MLP.

The block, with block 3's residual coefficients (block i's are in ``depth_scales``)::

    h = rmsnorm(x, input_layernorm)
    a = attention(h)                                         see ../_attention.py
    x = rmsnorm(a, post_attention_layernorm) / sqrt(8) + x * sqrt(7) / sqrt(8)
    h = rmsnorm(x, pre_feedforward_layernorm)
    m = moe(h)                                               routed scaling 2.5 included
    out = rmsnorm(m, post_feedforward_layernorm) / sqrt(9) + x * sqrt(8) / sqrt(9)

Blocks 3 and 4 attend through the 512-token sliding window with NeoX RoPE; block 5 is a
full-attention layer without position embedding: no rotation of q or k, the per-token
query scale on q, and the whole context as its window.

Every RMSNorm computes its statistics, normalization and affine in fp32 and rounds once;
each residual add rounds both scaled terms to bf16, then their sum. The residual
coefficients are the logical block's: block i's attention add is the 2(i + 1)-th add of
the residual stream, its feed-forward add the next one.

Logical layer and cache slot: ``layer_idx`` fixes the block's semantics; the cache slot is
the call's manager's local index for that layer -- 0 for a manager whose ``layer_mask``
selects only it, its own position in a manager holding several layers. A manager that
does not hold the layer -- a default one-layer manager holds layer 0 -- is rejected
before any kernel runs.

Host synchronization per call: one for the position check, and the MoE's dispatch
synchronizations (two, plus two per selected expert). Positions are absolute: a sliding
layer attends to the newest 512 keys of the request's history up to each query, the
full-attention layer to all of them.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.residual.beam_depth_residual import (
    beam_depth_residual,
    depth_scales,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata

from ..full_model_nvfp4__sm_103__tp4 import _attention
from ..moe_bf16__sm_103__tp1.moe import BeamMoeBf16Sm103Tp1, check_weights
from ..moe_bf16__sm_103__tp1.moe import check_config as check_moe_config

#: The configuration this component computes, as the checkpoint's config.json states it.
#: Anything else is rejected at construction rather than computed approximately; the
#: MoE component checks its own fields.
SUPPORTED_CONFIG = {
    "hidden_size": 5120,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "sliding_window": 512,
}
SUPPORTED_LAYER_IDX = 3
#: The sparse blocks of the checkpoint's schedule and the attention each one runs.
ATTENTION_TYPES = {3: "sliding_attention", 4: "sliding_attention", 5: "full_attention"}
SUPPORTED_LAYER_INDICES = tuple(ATTENTION_TYPES)
#: RoPE of the sliding layers; the full-attention layers have none.
SUPPORTED_ROPE = {"rope_type": "default", "rope_theta": 10000.0}

#: The most tokens one call may append: the six-layer model's sequence capacity. Longer
#: calls are rejected; the request's whole KV length is bounded by its cache's capacity.
MAX_TOKENS = 1024

_SM = (10, 3)

#: The MoE's tensors carry this prefix in the block's checkpoint names.
MLP_PREFIX = "mlp."


def weight_shapes(config) -> "OrderedDict[str, tuple[int, ...]]":
    """The block's twelve attention and norm tensors by checkpoint name, [out, in], all BF16."""
    hidden = config.hidden_size
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
            f"only sparse blocks {SUPPORTED_LAYER_INDICES} are supported, got layer_idx={layer_idx!r}"
        )
    for field, expected in SUPPORTED_CONFIG.items():
        actual = getattr(config, field, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"unsupported {field}={actual!r}; this block computes {expected!r}")
    attention_type = ATTENTION_TYPES[layer_idx]
    layer_types = getattr(config, "layer_types", None) or []
    if len(layer_types) <= layer_idx or layer_types[layer_idx] != attention_type:
        raise ValueError(
            f"block {layer_idx} must be a {attention_type} layer: "
            f"{layer_types[layer_idx] if len(layer_types) > layer_idx else None!r}"
        )
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope = rope_parameters.get(attention_type)
    expected_rope = SUPPORTED_ROPE if attention_type == "sliding_attention" else None
    if rope != expected_rope or attention_type not in rope_parameters:
        raise ValueError(
            f"unsupported {attention_type} rope parameters {rope!r}; need {expected_rope!r}"
        )
    check_moe_config(config, layer_idx)


class BeamSparseBlockBf16Sm103Tp1(nn.Module):
    """Beam sparse decoder block 3, 4 or 5: BF16 weights and activations, an FP32 router, one GPU,
    one request step.

    State: the twelve BF16 attention and norm tensors of ``weight_shapes`` and ``mlp``, the
    MoE component with its router and 256 experts; ``load_weights`` loads all 782.

    ``forward(hidden_states [1, T, 5120] bf16, position_ids [1, T], attn_metadata)``
    returns ``[1, T, 5120]`` bf16 for 1 <= T <= ``MAX_TOKENS``, a prepared
    ``TrtllmAttentionMetadata`` holding one step of one request on a BF16
    ``KVCacheManagerV2`` (8 kv heads of 128) that holds the block's logical layer -- a
    context step of T tokens after ``past`` cached ones (paged-context FMHA when
    ``past > 0``) or a one-token generation step -- and positions ``arange(past, past + T)``.
    Inputs outside that are rejected before any kernel runs; batches of several requests
    and other blocks are not computed here. Inputs and parameters are never modified.
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
        self.is_sliding = ATTENTION_TYPES[layer_idx] == "sliding_attention"
        self.window = config.sliding_window if self.is_sliding else None
        self.rope_theta = SUPPORTED_ROPE["rope_theta"] if self.is_sliding else None
        self.attention_scales = depth_scales(layer_idx, is_attention=True)
        self.feed_forward_scales = depth_scales(layer_idx, is_attention=False)
        self._shapes = weight_shapes(config)
        for name, shape in self._shapes.items():
            tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
            self.register_parameter(_attribute(name), nn.Parameter(tensor, requires_grad=False))
        self.mlp = BeamMoeBf16Sm103Tp1(config, layer_idx, device=device)
        self._loaded = False

    @property
    def hidden_size(self) -> int:
        return self._shapes["input_layernorm.weight"][0]

    def checkpoint_tensors(self) -> "OrderedDict[str, torch.Tensor]":
        """The block's 782 tensors by checkpoint name: its own twelve, then the MoE's."""
        tensors = OrderedDict((name, getattr(self, _attribute(name))) for name in self._shapes)
        for name, tensor in self.mlp.checkpoint_tensors().items():
            tensors[MLP_PREFIX + name] = tensor
        return tensors

    @torch.no_grad()
    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Copy exactly the block's 782 tensors, by checkpoint name, [out, in].

        The twelve attention and norm tensors are BF16; the ``mlp.``-prefixed MoE tensors
        are as ``BeamMoeBf16Sm103Tp1.load_weights`` describes (FP32 router, BF16 experts).
        Every tensor is checked before any parameter changes, so a rejected load leaves
        the block as it was. The block runs only after a load completed: a failure while
        copying leaves it unloaded.
        """
        if not isinstance(weights, Mapping):
            raise TypeError(f"weights must be a mapping, got {type(weights).__name__}")
        own = {n: w for n, w in weights.items() if not str(n).startswith(MLP_PREFIX)}
        moe = {n[len(MLP_PREFIX) :]: w for n, w in weights.items() if str(n).startswith(MLP_PREFIX)}
        check_weights(
            OrderedDict((n, (shape, torch.bfloat16)) for n, shape in self._shapes.items()),
            own,
            "block",
        )
        self.mlp.check_weights(moe)
        self._loaded = False
        device = self.input_layernorm.device
        staged = {name: own[name].to(device=device, copy=True) for name in self._shapes}
        for name, tensor in staged.items():
            getattr(self, _attribute(name)).copy_(tensor)
        self.mlp.load_weights(moe)
        self._loaded = True

    def _check_step(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, md: TrtllmAttentionMetadata
    ) -> None:
        """Raise ValueError on anything this block would otherwise compute wrongly."""
        if not self._loaded:
            raise ValueError("load_weights must be called before the block runs")
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
            window=self.window if self.is_sliding else step["max_seq_len"],
        )

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
        h = self.mlp(h)
        h = flashinfer_rmsnorm(h, self.post_feedforward_layernorm, self.eps)
        x = beam_depth_residual(h, x, *self.feed_forward_scales)
        return x.view(1, num_tokens, self.hidden_size)
