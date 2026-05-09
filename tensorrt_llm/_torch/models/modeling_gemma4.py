# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""TensorRT-LLM PyTorch backend implementation for Gemma4 text model."""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from packaging.version import Version
from torch import nn

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.modules.fused_moe.create_moe import create_moe
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.modules.fused_moe.routing import BaseMoeRoutingMethod
from tensorrt_llm._torch.modules.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping

from ..attention_backend import AttentionMetadata, FlashInferAttentionMetadata
from ..attention_backend.interface import (
    AttentionMask,
    CustomAttentionMask,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from ..modules.rms_norm import RMSNorm
from ..utils import ActivationType
from .modeling_utils import DecoderModel, DecoderModelForCausalLM, register_auto_model

_MIN_TRANSFORMERS_FOR_GEMMA4 = "5.5.0"
if Version(transformers.__version__) < Version(_MIN_TRANSFORMERS_FOR_GEMMA4):
    raise ImportError(
        f"Gemma4 requires transformers>={_MIN_TRANSFORMERS_FOR_GEMMA4}, "
        f"but found transformers=={transformers.__version__}. "
        f"Please upgrade: pip install 'transformers>={_MIN_TRANSFORMERS_FOR_GEMMA4}'"
    )

from transformers import Gemma4TextConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Scaled embedding (reused from Gemma3 pattern)
# ---------------------------------------------------------------------------
class Gemma4TextScaledWordEmbedding(Embedding):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: Optional[torch.dtype] = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,
        embed_scale: Optional[float] = None,
    ):
        super().__init__(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=tensor_parallel_mode,
            gather_output=gather_output,
        )
        if embed_scale is None:
            embed_scale = math.sqrt(hidden_size)
        self.embed_scale = torch.tensor(embed_scale, dtype=self.dtype)

    @torch.inference_mode()
    def forward(self, input_ids):
        return super().forward(input_ids) * self.embed_scale


# ---------------------------------------------------------------------------
# Activation helper (same as Gemma3)
# ---------------------------------------------------------------------------
def gelu_tanh(gate_x: torch.Tensor) -> torch.Tensor:
    if IS_FLASHINFER_AVAILABLE:
        return torch.ops.trtllm.flashinfer_gelu_tanh_and_mul(gate_x)
    gate, x = gate_x.chunk(2, dim=-1)
    return nn.functional.gelu(gate, approximate="tanh") * x


# ---------------------------------------------------------------------------
# Q-only linear for KV shared layers
# ---------------------------------------------------------------------------
class _QOnlyLinear(Linear):
    """Q-only linear with load_weights compatible with QKV fused path.

    Used by KV shared layers where HF doesn't create k/v projections.  Must
    be a TRT-LLM Linear so that out_features are sharded correctly under
    tensor parallelism (COLUMN mode).  The weight mapper routes the usual
    QKV list-of-three [q, k, v] dict-payload through ``load_weights``; the
    k/v entries are empty (HF does not emit them for shared layers) so we
    only consume index 0 and delegate to the vanilla TP-aware loader.
    """

    def load_weights(self, weights, allow_partial_loading=False):
        q_weights = weights[0] if isinstance(weights, list) else weights
        super().load_weights([q_weights], allow_partial_loading=allow_partial_loading)


# ---------------------------------------------------------------------------
# Gemma4 Attention
# ---------------------------------------------------------------------------
class Gemma4Attention(QKNormRoPEAttention):
    def __init__(
        self,
        model_config: ModelConfig[Gemma4TextConfig],
        layer_idx: Optional[int] = None,
        is_sliding: bool = False,
        is_kv_shared: bool = False,
        cache_layer_idx: Optional[int] = None,
    ):
        self.is_sliding = is_sliding
        self.is_kv_shared = is_kv_shared
        config = model_config.pretrained_config

        # Per-layer head_dim and kv heads
        # Note: num_global_key_value_heads is only used when K=V (alternative
        # attention). For non-K=V full layers, use regular num_key_value_heads.
        use_k_eq_v = getattr(config, "attention_k_eq_v", False) and not is_sliding
        if is_sliding:
            layer_head_dim = config.head_dim
            layer_num_kv_heads = config.num_key_value_heads
        else:
            layer_head_dim = getattr(config, "global_head_dim", config.head_dim)
            if use_k_eq_v:
                layer_num_kv_heads = (
                    getattr(config, "num_global_key_value_heads", None)
                    or config.num_key_value_heads
                )
            else:
                layer_num_kv_heads = config.num_key_value_heads

        # Build RoPE params per layer type
        rope_params = RopeParams()
        rope_params.max_positions = config.max_position_embeddings
        if is_sliding:
            # Sliding: default RoPE, theta=10K, full rotation
            rope_config = (
                config.rope_parameters.get("sliding_attention", {})
                if config.rope_parameters
                else {}
            )
            rope_params.theta = rope_config.get("rope_theta", 10_000.0)
            rope_params.scale_type = RotaryScalingType.none
            rope_params.scale = 1.0
            rope_params.dim = layer_head_dim
        else:
            # Full: proportional RoPE, theta=1M, partial_rotary_factor=0.25
            rope_config = (
                config.rope_parameters.get("full_attention", {}) if config.rope_parameters else {}
            )
            rope_params.theta = rope_config.get("rope_theta", 1_000_000.0)
            rope_params.scale_type = RotaryScalingType.none
            rope_params.scale = 1.0
            partial_rotary_factor = rope_config.get("partial_rotary_factor", 0.25)
            rope_params.dim = int(layer_head_dim * partial_rotary_factor)

        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=rope_params,
        )

        self.attention_window_size = None
        if is_sliding:
            self.attention_window_size = config.sliding_window

        # Gemma4 uses scaling=1.0 (no query_pre_attn_scalar).
        # Attention base: softmax_scale = 1 / (sqrt(head_dim) * q_scaling)
        # For scaling=1.0: q_scaling = 1 / sqrt(head_dim)
        q_scaling = 1.0 / math.sqrt(layer_head_dim)

        self.use_k_eq_v = use_k_eq_v

        # Temporarily override config.head_dim so the Attention base class
        # picks up the correct per-layer head_dim.
        original_head_dim = config.head_dim
        config.head_dim = layer_head_dim

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=layer_num_kv_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            q_scaling=q_scaling,
        )

        # Restore original config head_dim
        config.head_dim = original_head_dim

        # Fix proportional RoPE for full-attention layers.
        #
        # HF proportional RoPE produces cos/sin of shape [seq, head_dim] (512)
        # with only the first `rope_angles` (64) frequency pairs non-trivial;
        # the rest are zero-frequency (cos=1, sin=0).  Crucially, HF's
        # `rotate_half` splits at head_dim//2 = 256, pairing dim[i] with
        # dim[i+256].
        #
        # TRT-LLM's default RoPE with dim=128 only rotates the first 128
        # dimensions, pairing dim[i] with dim[i+64].  This is wrong because
        # the pairing dimension differs from HF (i+256 vs i+64).
        #
        # Fix: set rope dim = head_dim (512) so that rotate_half splits at
        # 256, matching HF. Pad inv_freq with zeros for non-rotary pairs.
        if not is_sliding and self.rotary_emb is not None:
            import numpy as np

            theta = rope_params.theta
            max_pos = rope_params.max_positions
            rope_angles = int(partial_rotary_factor * layer_head_dim // 2)
            half_dim = layer_head_dim // 2  # 256

            # Build inv_freq: first rope_angles (64) have real freq, rest zeros
            inv_freq_rotated = 1.0 / (
                theta ** (np.arange(0, 2 * rope_angles, 2, dtype=np.float32) / layer_head_dim)
            )
            inv_freq = np.concatenate(
                [
                    inv_freq_rotated,
                    np.zeros(half_dim - rope_angles, dtype=np.float32),
                ]
            )  # [256] — matches HF's proportional RoPE

            positions = np.arange(max_pos, dtype=np.float32)
            sinusoid = np.einsum("i,j->ij", positions, inv_freq)  # [max_pos, 256]
            cos_sin = np.stack([np.cos(sinusoid), np.sin(sinusoid)], axis=1)
            self.rotary_emb.rotary_cos_sin = torch.tensor(
                cos_sin, dtype=torch.float32, device="cuda"
            )
            # Update head_dim so apply_rotary_pos_emb uses full head_dim for
            # the rotate split, matching HF's rotate_half(head_dim//2) pairing.
            self.rotary_emb.head_dim = layer_head_dim

        # Use trtllm-gen for ALL layers.  trtllm-gen has pre-compiled cubins
        # for both H256+SWA and H512 across all supported dtypes.
        # For FP8 KV cache (NVFP4), Q is also cast to FP8 in the FlashInfer
        # backend so that QkvE4m3OBfloat16 context cubins can be used
        # (context cubins require same Q/KV dtype; decode cubins support
        # mixed dtypes natively).  Uniform backend avoids workspace
        # corruption between different wrapper types under CUDA graphs.
        self.attn.flashinfer_backend = "trtllm-gen"

        # KV shared layers: use target layer's index for KV cache access
        # so the attention backend reads from the target layer's cache slot.
        if cache_layer_idx is not None and cache_layer_idx != layer_idx:
            self.attn.layer_idx = cache_layer_idx

        # KV shared layers: replace fused QKV with Q-only projection.
        # HF doesn't create k/v for shared layers, so we match that.
        #
        # Reuse the parent's fused-QKV Linear's local Mapping (which has
        # ``tp_size=1`` under AttentionDP and ``tp_size=N`` under pure TP), so
        # the Q-only Linear follows the same sharding rule as q_proj inside
        # the parent's fused qkv_proj.
        if is_kv_shared:
            qkv_mapping = self.qkv_proj.mapping
            self.qkv_proj = _QOnlyLinear(
                in_features=config.hidden_size,
                out_features=qkv_mapping.tp_size * self.q_size,
                bias=False,
                dtype=config.torch_dtype,
                mapping=qkv_mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                weights_loading_config=WeightsLoadingConfig(weight_mode=WeightMode.VANILLA),
                quant_config=None,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                allreduce_strategy=model_config.allreduce_strategy,
            )

        # v_norm: always present in Gemma4 (no learnable scale).
        # For K=V layers, weight mapper duplicates k_proj→v_proj, so v equals
        # the raw k_proj output.  v_norm is applied to that raw value BEFORE
        # k_norm/RoPE — matching HF semantics: V = v_norm(k_proj(x)).
        # For regular layers, v_norm is applied to the v_proj output.
        self.v_norm = RMSNorm(
            hidden_size=layer_head_dim,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            has_weights=False,
        )

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: torch.Tensor,
    ):
        # Split QKV, apply QK norm
        if not self.fuse_qk_norm_rope:
            # KV shared layers: only q was produced (no k/v proj).  Apply
            # q_norm and rotary directly to q (Q-only fast path) without
            # the zero-padding / split_qkv / rope-on-zeros round-trip.
            if self.is_kv_shared:
                q_raw = self.q_norm(q.reshape(-1, self.head_dim)).reshape(-1, self.q_size)
                if not self.skip_rope and self.rotary_emb is not None:
                    # self.rotary_emb.forward supports a list of targets;
                    # pass only [q] to rotate Q in place.  The single-target
                    # path rejects the 2-target flashinfer fast path and
                    # falls through to the reference rotate_half
                    # implementation — correct regardless of head_dim.
                    [q_raw] = self.rotary_emb(position_ids, [q_raw])
                # Return None for k/v so FlashInfer skips cache append.
                return q_raw, None, None

            q, k, v = self.split_qkv(q, k, v)
            # For K=V layers, weight mapper duplicates k_proj weights into
            # v_proj, so after split_qkv v already equals k_proj(x).
            # HF order: v_norm is applied to the raw k_proj output, NOT
            # after k_norm.  So we must apply v_norm BEFORE qk_norm
            # mutates k.
            v = self.v_norm(v.reshape(-1, self.head_dim)).reshape(-1, self.kv_size)

            q, k = self.apply_qk_norm(q, k)

            if not self.skip_rope:
                return super(QKNormRoPEAttention, self).apply_rope(q, k, v, position_ids)
            else:
                return q, k, v

        # Fused path (not expected for Gemma4 since fuse_qk_norm_rope=False)
        qkv = q
        if k is not None and v is not None:
            qkv = torch.concat([q, k, v], dim=-1)
        return self.apply_qk_norm_rope(qkv, position_ids)

    def convert_qkv(self, q, k, v):
        """Override: KV-shared layers return q-only, skip convert/split."""
        if self.is_kv_shared and k is None and v is None:
            return q, None, None
        return super().convert_qkv(q, k, v)

    @torch.inference_mode()
    def forward(
        self,
        position_ids: Optional[torch.IntTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
        attention_mask_data: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if attention_mask_data is not None:
            assert isinstance(attn_metadata, FlashInferAttentionMetadata), (
                "Only FlashInfer backend supports custom attention mask currently."
            )
            assert attention_mask == CustomAttentionMask.CUSTOM
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            attention_window_size=self.attention_window_size,
            attention_mask_data=attention_mask_data,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Gemma4 MoE routing
# ---------------------------------------------------------------------------
class Gemma4Router(nn.Module):
    """Router for Gemma4 MoE.

    Applies: RMSNorm(no weight) -> scale * hidden_size^{-0.5} -> Linear proj
    to produce raw logits for expert selection.  Under TP/EP the router
    output must be computed on the FULL hidden state so every rank sees
    the full routing table before top-k dispatch; we therefore keep
    ``self.proj`` replicated (no ``tensor_parallel_mode``).
    """

    def __init__(self, model_config: ModelConfig[Gemma4TextConfig]):
        super().__init__()
        config = model_config.pretrained_config
        self.hidden_size = config.hidden_size
        self.scalar_root_size = self.hidden_size**-0.5

        self.norm = RMSNorm(
            hidden_size=self.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            has_weights=False,
        )
        self.scale = nn.Parameter(torch.ones(self.hidden_size, dtype=config.torch_dtype))
        self.proj = Linear(
            in_features=config.hidden_size,
            out_features=config.num_experts,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            quant_config=None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states * self.scale * self.scalar_root_size
        return self.proj(hidden_states)


class Gemma4MoeRoutingMethod(BaseMoeRoutingMethod):
    """Gemma4 routing: softmax -> topk -> renormalize -> per_expert_scale.

    The per_expert_scale is fetched at runtime via a callable so the parameter
    stays on the correct device.
    """

    def __init__(
        self,
        top_k: int,
        callable_per_expert_scale: callable,
        output_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.top_k = top_k
        self.callable_per_expert_scale = callable_per_expert_scale
        self.output_dtype = output_dtype

    def apply(self, router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # softmax over all experts
        router_probs = F.softmax(router_logits.to(self.output_dtype), dim=-1)
        # top-k selection
        topk_weights, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        # renormalize so selected weights sum to 1
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        # apply per-expert scale
        per_expert_scale = self.callable_per_expert_scale()
        expert_scales = per_expert_scale[topk_indices].to(topk_weights.dtype)
        topk_weights = topk_weights * expert_scales
        return topk_indices.to(torch.int32), topk_weights


class Gemma4MoE(nn.Module):
    """Gemma4 Mixture of Experts block.

    Contains a router (for logit computation) and fused experts
    dispatched via the custom Gemma4 routing method.
    """

    def __init__(
        self,
        model_config: ModelConfig[Gemma4TextConfig],
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.router = Gemma4Router(model_config)
        self.per_expert_scale = nn.Parameter(
            torch.ones(config.num_experts, dtype=config.torch_dtype)
        )

        routing_method = Gemma4MoeRoutingMethod(
            top_k=config.top_k_experts,
            callable_per_expert_scale=lambda: self.per_expert_scale,
        )

        self.experts = create_moe(
            routing_method=routing_method,
            num_experts=config.num_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            dtype=config.torch_dtype,
            reduce_results=True,
            model_config=model_config,
            layer_idx=layer_idx,
            activation_type=ActivationType.Geglu,
            # VANILLA mode: preprocess_weights splits 3D gate_up_proj into per-expert w1/w3
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_input: torch.Tensor,
        all_rank_num_tokens: Optional[list] = None,
    ) -> torch.Tensor:
        router_logits = self.router(router_input)
        return self.experts(
            hidden_states,
            router_logits,
            all_rank_num_tokens=all_rank_num_tokens,
        )


# ---------------------------------------------------------------------------
# Gemma4 Decoder Layer
# ---------------------------------------------------------------------------
class Gemma4DecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[Gemma4TextConfig],
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config

        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.is_sliding = is_sliding

        # Determine if this is a KV-shared layer
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0

        # For shared layers, find the target layer to read KV cache from:
        # last non-shared layer of the same attention type (sliding/full).
        cache_layer_idx = layer_idx
        if self.is_kv_shared_layer:
            non_shared_types = config.layer_types[:first_kv_shared_layer_idx]
            current_type = config.layer_types[layer_idx]
            if current_type in non_shared_types:
                cache_layer_idx = (
                    len(non_shared_types) - 1 - non_shared_types[::-1].index(current_type)
                )

        # Double-wide MLP for KV-shared layers when configured
        intermediate_size = config.intermediate_size
        if getattr(config, "use_double_wide_mlp", False) and self.is_kv_shared_layer:
            intermediate_size = config.intermediate_size * 2

        # Attention
        self.self_attn = Gemma4Attention(
            model_config,
            layer_idx=layer_idx,
            is_sliding=is_sliding,
            is_kv_shared=self.is_kv_shared_layer,
            cache_layer_idx=cache_layer_idx,
        )

        # Dense MLP.  Under AttentionDP each rank runs its own MLP on its
        # own tokens (no TP sharding), so we force ``overridden_tp_size=1``
        # which also disables the internal down_proj allreduce.  Under pure
        # TP we take the default (full tp_size) and rely on the down_proj
        # allreduce to produce a full-rank activation.
        mlp_tp_size = 1 if model_config.mapping.enable_attention_dp else None
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            activation=gelu_tanh,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            overridden_tp_size=mlp_tp_size,
        )

        # Layer norms (same pattern as Gemma3)
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.pre_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )
        self.post_feedforward_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        # Layer scalar
        self.register_buffer("layer_scalar", torch.ones(1))

        # MoE block (parallel with dense MLP)
        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        if self.enable_moe_block:
            self.moe = Gemma4MoE(model_config, layer_idx=layer_idx)
            self.post_feedforward_layernorm_1 = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )

        # Per-Layer Embedding (PLE) components in decoder layer
        self.hidden_size_per_layer_input = getattr(config, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            ple_dim = self.hidden_size_per_layer_input
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, ple_dim, bias=False, dtype=config.torch_dtype
            )
            self.per_layer_projection = nn.Linear(
                ple_dim, config.hidden_size, bias=False, dtype=config.torch_dtype
            )
            self.post_per_layer_input_norm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )

    @torch.inference_mode()
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        attention_mask_data: Optional[torch.Tensor] = None,
        per_layer_input: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # lora_params is handled explicitly by the MLP call below; drop it
        # from kwargs so it is not forwarded into self.self_attn (base
        # Attention.forward would raise on the extra kwarg).
        lora_params = kwargs.pop("lora_params", None)

        # Ensure dtype consistency (E2E warmup may pass float32)
        target_dtype = self.input_layernorm.weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)

        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=CustomAttentionMask.CUSTOM
            if attention_mask_data is not None
            else PredefinedAttentionMask.CAUSAL,
            attention_mask_data=attention_mask_data,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward (dense MLP + optional MoE in parallel)
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, lora_params=lora_params)

        if self.enable_moe_block:
            # MLP path: post-norm the MLP output
            hidden_states_mlp = self.post_feedforward_layernorm_1(hidden_states)
            # MoE path: router input is the residual (before MLP layernorm)
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            moe_input = self.pre_feedforward_layernorm_2(hidden_states_flat)
            # Under AttentionDP, the FusedMoE all-to-all / all-gather needs
            # each rank's token count; under pure TP (no ADP) this list is
            # None and FusedMoE falls back to x.shape[0].
            all_rank_num_tokens = getattr(attn_metadata, "all_rank_num_tokens", None)
            hidden_states_moe = self.moe(
                moe_input,
                hidden_states_flat,
                all_rank_num_tokens=all_rank_num_tokens,
            )
            hidden_states_moe = hidden_states_moe.reshape(residual.shape)
            hidden_states_moe = self.post_feedforward_layernorm_2(hidden_states_moe)
            # Combine MLP + MoE
            hidden_states = hidden_states_mlp + hidden_states_moe

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # Per-Layer Embedding (PLE) injection
        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            gate = F.gelu(self.per_layer_input_gate(hidden_states), approximate="tanh")
            gated = gate * per_layer_input
            projected = self.per_layer_projection(gated)
            hidden_states = self.post_per_layer_input_norm(projected) + residual

        # Layer scalar
        hidden_states = hidden_states * self.layer_scalar

        return hidden_states


# ---------------------------------------------------------------------------
# Gemma4 Text Model
# ---------------------------------------------------------------------------
class Gemma4TextModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[Gemma4TextConfig]):
        super().__init__(model_config)
        config = self.model_config
        pretrained = config.pretrained_config
        self.hidden_size = pretrained.hidden_size

        # Under AttentionDP, each rank runs the full sequence locally so the
        # embedding must be replicated (tp_size=1 effectively); otherwise
        # embed_tokens is COLUMN-sharded by vocab.  This mirrors the
        # tied-lm_head assertion in DecoderModelForCausalLM.__init__ which
        # requires embed_tokens.tp_size == lm_head.tp_size (lm_head drops to
        # tp_size=1 under ADP).
        if config.mapping.enable_attention_dp:
            self.embed_tokens = Gemma4TextScaledWordEmbedding(
                pretrained.vocab_size,
                pretrained.hidden_size,
                dtype=pretrained.torch_dtype,
                embed_scale=math.sqrt(pretrained.hidden_size),
            )
        else:
            self.embed_tokens = Gemma4TextScaledWordEmbedding(
                pretrained.vocab_size,
                pretrained.hidden_size,
                dtype=pretrained.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
                embed_scale=math.sqrt(pretrained.hidden_size),
            )

        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(model_config, layer_idx)
                for layer_idx in range(pretrained.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=pretrained.hidden_size,
            eps=pretrained.rms_norm_eps,
            dtype=pretrained.torch_dtype,
        )

        # Per-Layer Embedding (PLE) model-level components
        self.hidden_size_per_layer_input = getattr(pretrained, "hidden_size_per_layer_input", 0)
        if self.hidden_size_per_layer_input:
            ple_dim = self.hidden_size_per_layer_input
            num_layers = pretrained.num_hidden_layers
            vocab_size_ple = getattr(
                pretrained, "vocab_size_per_layer_input", pretrained.vocab_size
            )

            if config.mapping.enable_attention_dp:
                self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                    vocab_size_ple,
                    num_layers * ple_dim,
                    dtype=pretrained.torch_dtype,
                    embed_scale=math.sqrt(ple_dim),
                )
            else:
                self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                    vocab_size_ple,
                    num_layers * ple_dim,
                    dtype=pretrained.torch_dtype,
                    mapping=config.mapping,
                    tensor_parallel_mode=TensorParallelMode.COLUMN,
                    gather_output=True,
                    embed_scale=math.sqrt(ple_dim),
                )

            self.per_layer_model_projection = nn.Linear(
                pretrained.hidden_size,
                num_layers * ple_dim,
                bias=False,
                dtype=pretrained.torch_dtype,
            )
            self.per_layer_model_projection_scale = pretrained.hidden_size**-0.5
            self.per_layer_projection_norm = RMSNorm(
                hidden_size=ple_dim,
                eps=pretrained.rms_norm_eps,
                dtype=pretrained.torch_dtype,
            )
            self.per_layer_input_scale = 2.0**-0.5

    def _compute_per_layer_inputs(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute combined per-layer embeddings + projections."""
        if not self.hidden_size_per_layer_input:
            return None
        if input_ids is None:
            return None

        config = self.model_config.pretrained_config
        num_layers = config.num_hidden_layers
        ple_dim = self.hidden_size_per_layer_input

        # Per-layer token embeddings: [N, num_layers * ple_dim]
        per_layer_embed = self.embed_tokens_per_layer(input_ids)
        per_layer_embed = per_layer_embed.reshape(-1, num_layers, ple_dim).to(self.dtype)

        # Project main embeddings: [N, hidden_size] -> [N, num_layers * ple_dim]
        projection = (
            self.per_layer_model_projection(
                inputs_embeds.to(self.per_layer_model_projection.weight.dtype)
            )
            * self.per_layer_model_projection_scale
        )
        projection = projection.to(self.dtype).reshape(-1, num_layers, ple_dim)
        projection = self.per_layer_projection_norm(projection)

        # Combine and scale
        per_layer_inputs = (projection + per_layer_embed) * self.per_layer_input_scale
        return per_layer_inputs

    def _ensure_ple_dtype(self):
        """Ensure PLE nn.Linear modules match model dtype (they may be
        float32 if the weight loader didn't handle nn.Linear correctly)."""
        if not self.hidden_size_per_layer_input:
            return
        target = self.dtype
        for mod in [self.per_layer_model_projection, self.per_layer_projection_norm]:
            if hasattr(mod, "weight") and mod.weight.dtype != target:
                mod.to(target)
        for layer in self.layers:
            for name in [
                "per_layer_input_gate",
                "per_layer_projection",
                "post_per_layer_input_norm",
            ]:
                mod = getattr(layer, name, None)
                if mod is not None and hasattr(mod, "weight") and mod.weight.dtype != target:
                    mod.to(target)

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        local_attention_mask_data: Optional[torch.Tensor] = None,
        global_attention_mask_data: Optional[torch.Tensor] = None,
        ple_input_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds.to(self.dtype)

        # Compute PLE inputs.  When both input_ids and inputs_embeds are
        # provided (multimodal path, where inputs_embeds has audio/image
        # features scattered in), use a PLE-safe ``ple_input_ids`` with the
        # multimodal token IDs replaced by pad_token_id so the per-layer
        # embedding table is not consulted at multimodal positions.  This
        # matches the HF Gemma4Model behaviour and is required for E2B/E4B
        # (which rely on PLE) to produce coherent output for multimodal
        # requests.
        ple_ids = ple_input_ids if ple_input_ids is not None else input_ids
        per_layer_inputs = self._compute_per_layer_inputs(ple_ids, hidden_states)

        for i, decoder_layer in enumerate(self.layers):
            per_layer_input = per_layer_inputs[:, i, :] if per_layer_inputs is not None else None

            hidden_states = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                attention_mask_data=local_attention_mask_data
                if decoder_layer.is_sliding
                else global_attention_mask_data,
                per_layer_input=per_layer_input,
                **kwargs,
            )

        if hidden_states.dtype != self.dtype:
            hidden_states = hidden_states.to(self.dtype)
        hidden_states = self.norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Gemma4 For Causal LM
# ---------------------------------------------------------------------------
@register_auto_model("Gemma4ForCausalLM")
class Gemma4ForCausalLM(DecoderModelForCausalLM[Gemma4TextModel, Gemma4TextConfig]):
    def __init__(
        self,
        model_config: ModelConfig[Gemma4TextConfig],
    ):
        mapping = model_config.mapping
        assert mapping.pp_size == 1, (
            "Gemma4 does not support pipeline parallelism "
            "(layer_scalar broadcast + hybrid VSWA pool pinning not yet implemented); "
            "use tensor_parallel_size or attention_dp instead."
        )
        assert mapping.cp_size == 1, (
            "Gemma4 does not support context parallelism "
            "(conflicts with per-layer head_dim + VSWA)."
        )
        if mapping.moe_ep_size > 1:
            assert getattr(model_config.pretrained_config, "enable_moe_block", False), (
                "moe_ep_size>1 requires a Gemma4 MoE variant (only 26B-A4B-it today)."
            )

        super().__init__(
            Gemma4TextModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        """Gemma4-specific defaults.

        FlashInfer backend is required for hybrid attention (per-layer
        head_dim 256/512 with VSWA), trtllm-gen cubin dispatch, and
        bidirectional attention masks for multimodal tokens.
        """
        return {
            "attn_backend": "FLASHINFER",
        }

    def _get_token_type_mask(self, mm_token_type_ids: torch.Tensor):
        """Build bidirectional attention mask from mm_token_type_ids.

        mm_token_type_ids: 0=text, 1=image, 2=video (or any positive int for
        a modality blob). Tokens within the same contiguous blob of the same
        modality attend bidirectionally to each other.
        """
        device = mm_token_type_ids.device
        token_type_ids = mm_token_type_ids.clone()
        # We only care about non-zero (multimodal) tokens
        is_mm = token_type_ids > 0

        # Detect blob boundaries: positions where type changes or goes 0->nonzero
        padded = torch.cat(
            (torch.tensor([0], device=device, dtype=token_type_ids.dtype), token_type_ids)
        )
        # A new blob starts whenever the type changes and the current is nonzero
        changes = (padded[1:] != padded[:-1]) & is_mm
        blob_ids = torch.cumsum(changes.int(), dim=0)
        # Mask out text tokens
        blob_ids = blob_ids * is_mm.int()

        # Tokens with the same non-zero blob_id attend bidirectionally
        token_type_mask = blob_ids.unsqueeze(0) == blob_ids.unsqueeze(1)
        # Only apply bidirectional mask for multimodal tokens
        token_type_mask = torch.where(blob_ids == 0, False, token_type_mask)

        return token_type_mask

    def get_context_mask(
        self,
        mm_token_type_ids: torch.Tensor,
        effective_sliding_window: Optional[int] = None,
    ):
        """Build context mask with causal + bidirectional for MM tokens."""
        device = mm_token_type_ids.device
        sequence_length = len(mm_token_type_ids)

        if effective_sliding_window is None or effective_sliding_window >= sequence_length:
            causal_mask = torch.arange(sequence_length, device=device).unsqueeze(0) <= torch.arange(
                sequence_length, device=device
            ).unsqueeze(1)
        else:
            pos = torch.arange(sequence_length, device=device)
            causal_mask = (pos.unsqueeze(0) <= pos.unsqueeze(1)) & (
                pos.unsqueeze(0) > pos.unsqueeze(1) - effective_sliding_window
            )

        token_type_mask = self._get_token_type_mask(mm_token_type_ids)
        causal_mask = causal_mask.masked_fill(token_type_mask, True)
        return causal_mask

    def get_flashinfer_attention_mask(
        self,
        mm_token_type_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        effective_sliding_window: Optional[int] = None,
    ) -> torch.Tensor:
        """Build FlashInfer custom mask for context requests."""
        assert isinstance(attn_metadata, FlashInferAttentionMetadata), (
            "Only FlashInfer backend supports custom mask currently."
        )
        num_contexts = attn_metadata.num_contexts
        assert num_contexts > 0

        qo_indptr = attn_metadata.qo_indptr[: num_contexts + 1]
        cached_token_lens = attn_metadata.cached_token_lens[:num_contexts]
        assert (cached_token_lens == 0).all()

        context_mask_list = []
        for i in range(num_contexts):
            mask_i = self.get_context_mask(
                mm_token_type_ids=mm_token_type_ids[qo_indptr[i] : qo_indptr[i + 1]],
                effective_sliding_window=effective_sliding_window,
            )
            context_mask_list.append(mask_i.flatten())
        return torch.cat(context_mask_list, dim=0).contiguous()

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        return_context_logits: bool = False,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        local_attention_mask_data = None
        global_attention_mask_data = None
        # Only build bidirectional masks when use_bidirectional_attention is
        # set to "vision" (26B, 31B).  E2B/E4B have this as None and should
        # use standard causal attention even for multimodal tokens.
        use_bidir = getattr(self.config, "use_bidirectional_attention", None)
        if mm_token_type_ids is not None and use_bidir == "vision":
            global_attention_mask_data = self.get_flashinfer_attention_mask(
                mm_token_type_ids=mm_token_type_ids,
                attn_metadata=attn_metadata,
                effective_sliding_window=None,
            )
            local_attention_mask_data = self.get_flashinfer_attention_mask(
                mm_token_type_ids=mm_token_type_ids,
                attn_metadata=attn_metadata,
                effective_sliding_window=self.config.sliding_window,
            )

        output = self.model(
            input_ids=input_ids,
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            local_attention_mask_data=local_attention_mask_data,
            global_attention_mask_data=global_attention_mask_data,
            ple_input_ids=kwargs.pop("ple_input_ids", None),
            **kwargs,
        )

        logits = self.logits_processor.forward(
            output,
            self.lm_head,
            attn_metadata,
            return_context_logits,
        )

        # Logit softcapping
        if self.config.final_logit_softcapping is not None:
            cap = self.config.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap

        return logits

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(weights, weight_mapper)
        # Ensure PLE nn.Linear modules match model dtype (weight loader may
        # not handle raw nn.Linear correctly, leaving them as float32).
        self.model._ensure_ple_dtype()
