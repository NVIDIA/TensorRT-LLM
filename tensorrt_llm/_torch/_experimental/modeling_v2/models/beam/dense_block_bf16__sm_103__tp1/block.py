# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam decoder block 0 / sm_103 / tp1: one dense, sliding-attention block, constructed directly.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no ``ForCausalLM``: nothing in the engine builds this class, and it serves no request.
A caller constructs it from the checkpoint's configuration, loads the block's fifteen
tensors and runs one fresh BF16 context through it, with the attention batch state
prepared on a ``KVCacheManagerV2``. The forward is assembled from catalog entries and
this component's own ``ops/``, and it does not call the built-in Beam modeling.

The block::

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

import math
from collections import OrderedDict
from typing import Mapping

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (  # noqa: E501
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    STANDARD,
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm.bindings import DataType

from .ops.attention.beam_neox_rope import beam_neox_rope
from .ops.attention.beam_output_gate import beam_output_gate
from .ops.residual.beam_depth_residual import beam_depth_residual

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
SUPPORTED_ROPE = {"rope_type": "default", "rope_theta": 10000.0}

#: The longest fresh context the block was measured at; longer ones are rejected.
MAX_TOKENS = 129

_SM = (10, 3)

#: thop_attention arguments that do not change with the step on this block: packed q|k|v,
#: append-and-attend, causal, bf16 pool, no quantization, and RoPE applied before the
#: call -- the rope_* values are inert placeholders the op requires, not this model's RoPE.
_CALL_CONSTANTS = dict(
    k=None,
    v=None,
    output_sf=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    attention_input_type=STANDARD,
    is_mla_enable=False,
    mask_type=1,
    q_scaling=1.0,
    quant_mode=0,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    attention_sinks=None,
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    out_scale=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    chunked_prefill_buffer_batch_size=1,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)


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


def depth_scales(layer_idx: int, is_attention: bool) -> tuple[float, float]:
    """(branch, residual) scales of a block's attention or feed-forward residual add.

    The n-th add of the residual stream scales the branch by 1 / sqrt(n) and the stream
    by sqrt(n - 1) / sqrt(n); block i's attention add is the 2(i + 1)-th, its
    feed-forward add the next one.
    """
    num_adds = 2 * (layer_idx + 1) + (0 if is_attention else 1)
    denom = math.sqrt(num_adds)
    return 1.0 / denom, math.sqrt(num_adds - 1) / denom


def check_config(config, layer_idx: int) -> None:
    """Raise ValueError naming the first field outside what this component computes."""
    if layer_idx != SUPPORTED_LAYER_IDX:
        raise ValueError(
            f"only block {SUPPORTED_LAYER_IDX} is supported, got layer_idx={layer_idx}"
        )
    for field, expected in SUPPORTED_CONFIG.items():
        actual = getattr(config, field, None)
        if actual != expected or type(actual) is not type(expected):
            raise ValueError(f"unsupported {field}={actual!r}; this block computes {expected!r}")
    layer_types = getattr(config, "layer_types", None) or []
    if len(layer_types) <= layer_idx or layer_types[layer_idx] != "sliding_attention":
        raise ValueError(
            f"block {layer_idx} must be a sliding_attention layer: {layer_types[:1]!r}"
        )
    mlp_types = getattr(config, "mlp_layer_types", None) or []
    if len(mlp_types) <= layer_idx or mlp_types[layer_idx] != "dense":
        raise ValueError(f"block {layer_idx} must have a dense MLP: {mlp_types[:1]!r}")
    rope = (getattr(config, "rope_parameters", None) or {}).get("sliding_attention")
    if rope != SUPPORTED_ROPE:
        raise ValueError(
            f"unsupported sliding_attention rope parameters {rope!r}; need {SUPPORTED_ROPE!r}"
        )


class BeamDenseBlockBf16Sm103Tp1(nn.Module):
    """Beam decoder block 0: BF16 weights and activations, one GPU, one fresh context.

    ``forward(hidden_states [1, T, 5120] bf16, position_ids [1, T], attn_metadata)``
    returns ``[1, T, 5120]`` bf16 for 1 <= T <= 129, positions ``arange(T)`` and a
    prepared ``TrtllmAttentionMetadata`` holding exactly that one context request on a
    single-layer BF16 ``KVCacheManagerV2`` (8 kv heads of 128, no cached prefix). Inputs
    outside that are rejected before any kernel runs; decode, cached prefixes, chunked
    prefill, batches and other blocks are not computed here.
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

    def checkpoint_tensors(self) -> "OrderedDict[str, torch.Tensor]":
        """The block's tensors by checkpoint name, in checkpoint order."""
        return OrderedDict((name, getattr(self, _attribute(name))) for name in self._shapes)

    @torch.no_grad()
    def load_weights(self, weights: Mapping[str, torch.Tensor]) -> None:
        """Copy exactly the block's fifteen tensors, by checkpoint name, BF16, [out, in]."""
        missing = [n for n in self._shapes if n not in weights]
        unexpected = sorted(n for n in weights if n not in self._shapes)
        if missing or unexpected:
            raise ValueError(f"block weights mismatch: missing {missing}, unexpected {unexpected}")
        for name, shape in self._shapes.items():
            tensor = weights[name]
            if tuple(tensor.shape) != shape or tensor.dtype != torch.bfloat16:
                raise ValueError(
                    f"{name}: got {tuple(tensor.shape)} {tensor.dtype}, need {shape} torch.bfloat16"
                )
            getattr(self, _attribute(name)).copy_(tensor)

    def _check_step(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, md: TrtllmAttentionMetadata
    ) -> None:
        """Raise ValueError on anything this block would otherwise compute wrongly."""
        if not isinstance(hidden_states, torch.Tensor) or hidden_states.dtype != torch.bfloat16:
            raise ValueError("hidden_states must be a torch.bfloat16 tensor")
        if (
            hidden_states.dim() != 3
            or hidden_states.shape[0] != 1
            or hidden_states.shape[2] != self.hidden_size
        ):
            raise ValueError(
                f"hidden_states must be [1, T, {self.hidden_size}], got {tuple(hidden_states.shape)}"
            )
        if hidden_states.device != self.input_layernorm.device or not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous on the block's device")
        num_tokens = hidden_states.shape[1]
        if not 1 <= num_tokens <= MAX_TOKENS:
            raise ValueError(f"supported context lengths are 1..{MAX_TOKENS}, got {num_tokens}")
        if (
            not isinstance(position_ids, torch.Tensor)
            or position_ids.is_floating_point()
            or tuple(position_ids.shape) != (1, num_tokens)
            or position_ids.device != hidden_states.device
        ):
            raise ValueError(
                f"position_ids must be an integer [1, {num_tokens}] tensor on the block's device"
            )
        expected = torch.arange(num_tokens, dtype=position_ids.dtype, device=position_ids.device)
        if not torch.equal(position_ids[0], expected):
            raise ValueError("a fresh context must have positions 0..T-1")
        if not isinstance(md, TrtllmAttentionMetadata):
            raise ValueError(
                f"attn_metadata must be TrtllmAttentionMetadata, got {type(md).__name__}"
            )
        mgr = md.kv_cache_manager
        if mgr is None:
            raise ValueError("attn_metadata needs a KV cache manager for the paged append")
        if md.num_seqs != 1 or md.num_contexts != 1 or md.num_ctx_tokens != num_tokens:
            raise ValueError(
                f"one context request of {num_tokens} tokens is supported; metadata has "
                f"{md.num_seqs} sequences, {md.num_contexts} contexts, {md.num_ctx_tokens} context tokens"
            )
        if (
            int(md.kv_lens_runtime[0]) != num_tokens
            or int(md.prompt_lens_cpu_runtime[0]) != num_tokens
        ):
            raise ValueError(
                "a cached prefix is not supported: the request must start at an empty cache"
            )
        if (
            md.use_paged_context_fmha
            or md.is_cross
            or md.is_spec_decoding_enabled
            or md.beam_width != 1
        ):
            raise ValueError(
                "paged-context FMHA, cross attention, speculative decoding and beams are not supported"
            )
        heads = getattr(mgr, "num_kv_heads_per_layer", None) or [None]
        if (
            mgr.dtype != DataType.BF16
            or heads[self.layer_idx] != self.num_kv_heads
            or mgr.head_dim != self.head_dim
        ):
            raise ValueError(
                f"the KV cache must be BF16 with {self.num_kv_heads} kv heads of {self.head_dim}; "
                f"got {mgr.dtype}, {heads}, head_dim {mgr.head_dim}"
            )

    @property
    def hidden_size(self) -> int:
        return self._shapes["input_layernorm.weight"][0]

    @staticmethod
    def step_args(md: TrtllmAttentionMetadata) -> dict:
        """Project the prepared metadata onto thop_attention's explicit batch state."""
        return dict(
            sequence_length=md.kv_lens_cuda_runtime,
            host_past_key_value_lengths=md.kv_lens_runtime,
            host_total_kv_lens=md.host_total_kv_lens,
            context_lengths=md.prompt_lens_cuda_runtime,
            host_context_lengths=md.prompt_lens_cpu_runtime,
            host_request_types=md.host_request_types_runtime,
            kv_cache_block_offsets=md.kv_cache_block_offsets,
            host_kv_cache_pool_pointers=md.host_kv_cache_pool_pointers,
            host_kv_cache_pool_mapping=md.host_kv_cache_pool_mapping,
            workspace_=md.effective_workspace,
            tokens_per_block=md.tokens_per_block,
            max_num_requests=md.max_num_requests,
            max_context_length=md.max_context_length,
            max_seq_len=md.max_seq_len,
            num_contexts=md.num_contexts,
            num_ctx_tokens=md.num_ctx_tokens,
            trtllm_gen_jit_warmup=md.trtllm_gen_jit_warmup,
            use_paged_context_fmha=md.use_paged_context_fmha,
            beam_width=md.effective_beam_width,
            cache_indirection=md.cache_indirection,
            block_ids_per_seq=md.block_ids_per_seq,
            max_context_q_len_override=md.max_context_q_len_override,
            is_cross=md.is_cross,
            is_spec_decoding_enabled=md.is_spec_decoding_enabled,
            use_spec_decoding=md.use_spec_decoding,
            is_spec_dec_tree=md.is_spec_dec_tree,
            spec_decoding_generation_lengths=md.spec_decoding_generation_lengths,
            spec_decoding_position_offsets_for_cpp=md.spec_decoding_position_offsets_for_cpp,
            spec_decoding_packed_mask=md.spec_decoding_packed_mask,
            spec_decoding_bl_tree_mask_offset=md.spec_decoding_bl_tree_mask_offset,
            spec_decoding_bl_tree_mask=md.spec_decoding_bl_tree_mask,
            spec_decoding_target_max_draft_tokens=md.max_total_draft_tokens,
            spec_bl_tree_first_sparse_mask_offset_kv=md.spec_bl_tree_first_sparse_mask_offset_kv,
            num_sparse_topk=md.num_sparse_topk,
            flash_mla_tile_scheduler_metadata=md.flash_mla_tile_scheduler_metadata,
            flash_mla_num_splits=md.flash_mla_num_splits,
            max_num_sequences=md.max_num_sequences,
            force_prepare_spec_dec_tree_mask=md.force_prepare_spec_dec_tree_mask,
        )

    def attention(self, h: torch.Tensor, positions: torch.Tensor, step: dict) -> torch.Tensor:
        """Normalized input [T, hidden] -> attention output [T, hidden], before post_attention_layernorm."""
        num_tokens = h.shape[0]
        heads, kv_heads, d = self.num_heads, self.num_kv_heads, self.head_dim
        q = cublas_mm(h, self.self_attn_q_proj.t())
        k = cublas_mm(h, self.self_attn_k_proj.t())
        v = cublas_mm(h, self.self_attn_v_proj.t())
        gate = cublas_mm(h, self.self_attn_gate_proj.t())
        q = flashinfer_rmsnorm(q.view(num_tokens * heads, d), self.self_attn_q_norm, self.eps)
        k = flashinfer_rmsnorm(k.view(num_tokens * kv_heads, d), self.self_attn_k_norm, self.eps)
        q, k = beam_neox_rope(
            q.view(num_tokens, heads, d),
            k.view(num_tokens, kv_heads, d),
            positions,
            self.rope_theta,
            1.0,
        )
        qkv = torch.cat(
            (q.view(num_tokens, heads * d), k.view(num_tokens, kv_heads * d), v), dim=-1
        )
        core = torch.empty(num_tokens, heads * d, dtype=h.dtype, device=h.device)
        core = thop_attention(
            q=qkv,
            output=core,
            local_layer_idx=self.layer_idx,
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_size=d,
            attention_window_size=self.window,
            **step,
            **_CALL_CONSTANTS,
        )
        gated = beam_output_gate(
            core.view(num_tokens, heads, d),
            gate.view(num_tokens, heads, d),
            self.self_attn_gate_bias,
        )
        return cublas_mm(gated.view(num_tokens, heads * d), self.self_attn_o_proj.t())

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
