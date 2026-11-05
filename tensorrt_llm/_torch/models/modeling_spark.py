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
"""XHToken Spark-X2.5/X2.6 (``Spark2_5ForCausalLM``) for the PyTorch backend.

Dense decoder with a 3:1 hybrid of sliding-window (512) and full-attention
layers driven by ``config.layer_types``. Full-attention layers use partial
RoPE (``partial_rotary_factor`` of the per-layer-type ``rope_parameters``
entry); every layer applies a headwise sigmoid gate (``g_proj``) to the
attention output before the output projection. The architecture is shared by
Spark-X2.5 and Spark-X2.6; sizes differ per checkpoint.
"""

from typing import Any, Dict, Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.configs.spark import Spark2_5Config
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType

from ..attention.attention import Attention
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PositionalEmbeddingParams, RopeParams
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import inference_mode_unless_compiling
from .checkpoints.hf.weight_mapper import HfWeightMapper
from .modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    register_auto_model,
    register_mapper,
)

_GATE_ACT_FNS = {
    "sigmoid": torch.sigmoid,
    "silu": F.silu,
}


def _apply_output_gate(
    attn_output: torch.Tensor,
    gate: torch.Tensor,
    num_heads: int,
    head_dim: int,
    act_mode: str = "sigmoid",
) -> torch.Tensor:
    """Apply a headwise gate to packed attention output.

    Args:
        attn_output: ``[num_tokens, num_heads * head_dim]`` attention kernel
            output, before the output projection.
        gate: ``[num_tokens, num_heads]`` raw gate logits (the ``g_proj``
            output); one scalar per head, broadcast over ``head_dim``.
        num_heads: local query-head count.
        head_dim: per-head dimension.
        act_mode: ``"sigmoid"`` or ``"silu"``
            (``config.gate_attn_act_mode``).

    The activation is computed in float32 and cast back to ``attn_output``'s
    dtype, matching the HF reference implementation.
    """
    try:
        act_fn = _GATE_ACT_FNS[act_mode]
    except KeyError:
        raise ValueError(
            f"Unsupported gate_attn_act_mode: {act_mode!r}; expected one of {sorted(_GATE_ACT_FNS)}"
        ) from None
    gate = act_fn(gate.float()).to(attn_output.dtype).unsqueeze(-1)
    shape = attn_output.shape
    return (attn_output.view(*shape[:-1], num_heads, head_dim) * gate).view(shape)


def _build_rope_params(
    config: Spark2_5Config,
    layer_type: str,
    max_positions: Optional[int] = None,
) -> RopeParams:
    """Build RoPE parameters for one layer type.

    Spark keeps a per-layer-type RoPE dict (``rope_parameters[layer_type]``
    with ``rope_theta`` / ``partial_rotary_factor``). The partial factor
    scales the rotary dimension only: frequencies are built over
    ``int(head_dim * partial_rotary_factor)`` and the remaining head
    dimensions pass through unrotated (neox-style rotate-half within the
    rotary slice), which matches ``RopeParams.dim`` semantics.
    """
    if max_positions is None:
        max_positions = config.max_position_embeddings
    return RopeParams(
        dim=int(config.head_dim * config.get_partial_rotary_factor(layer_type)),
        theta=float(config.get_rope_theta(layer_type)),
        scale_type=RotaryScalingType.none,
        scale=1.0,
        max_positions=max_positions,
    )


def gelu_erf(gate_x: torch.Tensor) -> torch.Tensor:
    """GatedMLP activation for Spark's ``down(erf-gelu(gate) * up)`` MLP."""
    gate, x = gate_x.chunk(2, dim=-1)
    return F.gelu(gate) * x


class Spark2_5Attention(Attention):
    """Spark attention: per-layer-type RoPE, optional sliding window, and a
    headwise gate (``g_proj``) applied to the attention output before
    ``o_proj``.

    The gate rides the base class's output-gate hooks -- ``preprocess_qkv``
    returns it alongside q/k/v and ``forward_impl`` applies it through the
    overridden ``apply_output_gate`` -- so the full ``Attention.forward``
    feature set (LoRA on QKV, dynamic-tree spec-dec position adjustment,
    attention sinks, relative attention bias, sparse hooks, the GQA-decode
    KV-append plan, helix context parallelism) is preserved.
    """

    def __init__(
        self,
        model_config: ModelConfig[Spark2_5Config],
        layer_idx: Optional[int] = None,
        reduce_output: bool = True,
    ):
        config = model_config.pretrained_config

        layer_type = config.layer_types[layer_idx]
        self.attention_window_size = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        # RoPE is per layer type (theta and partial rotary factor differ
        # between sliding and full layers). Attention backends read the rope
        # parameters from this layer's own PositionalEmbeddingParams, so the
        # fusion decision is left to the backend (same approach as Gemma3's
        # per-layer theta).
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=_build_rope_params(config, layer_type),
        )

        self.gate_act_mode = config.gate_attn_act_mode
        self.use_headwise_gate = bool(config.headwise_attn_output_gate)
        self._pending_gate = None

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
            reduce_output=reduce_output,
            head_dim=config.head_dim,
        )

        self.g_proj = None
        if self.use_headwise_gate:
            # In DEP mode (enable_attention_dp=True) the attention base class
            # uses a local mapping with tp_size=1, so self.num_heads is the
            # full head count and g_proj must not be sharded either.
            g_tp_mode = (
                None if model_config.mapping.enable_attention_dp else TensorParallelMode.COLUMN
            )
            self.g_proj = Linear(
                config.hidden_size,
                config.num_attention_heads,
                bias=config.attention_bias,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=g_tp_mode,
                quant_config=model_config.get_quant_config(),
            )

    def forward(self, position_ids, hidden_states, *args, **kwargs):
        # Capture the gate from the same (post-input_layernorm) hidden_states
        # that feed the QKV projection, then delegate to the base forward.
        # preprocess_qkv picks the stashed gate up and forward_impl applies it
        # via apply_output_gate before the output projection.
        if self.use_headwise_gate:
            self._pending_gate = self.g_proj(hidden_states)
        if kwargs.get("attention_window_size") is None and self.attention_window_size is not None:
            kwargs["attention_window_size"] = self.attention_window_size
        try:
            return super().forward(position_ids, hidden_states, *args, **kwargs)
        finally:
            self._pending_gate = None

    def preprocess_qkv(
        self,
        qkv: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: Optional[AttentionMetadata] = None,
    ):
        q, k, v, gate, kv_append = super().preprocess_qkv(qkv, position_ids, attn_metadata)
        if gate is None:
            gate = self._pending_gate
        return q, k, v, gate, kv_append

    def apply_output_gate(self, attention_output: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return _apply_output_gate(
            attention_output, gate, self.num_heads, self.head_dim, self.gate_act_mode
        )

    def _use_quantize_output(self):
        # A pre-quantized attention output (fp8, or an Fp4QuantizedTensor)
        # cannot carry the headwise gate math; opt out the same way the base
        # class does for attn_output_gate.
        if self.use_headwise_gate:
            return False
        return super()._use_quantize_output()


class Spark2_5DecoderLayer(DecoderLayer):
    def __init__(self, model_config: ModelConfig[Spark2_5Config], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.mapping = model_config.mapping

        self.self_attn = Spark2_5Attention(
            model_config,
            layer_idx=layer_idx,
            reduce_output=self.mapping.tp_size > 1,
        )

        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=config.mlp_bias,
            activation=gelu_erf,
            dtype=config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    @inference_mode_unless_compiling
    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)

        return hidden_states, residual


class Spark2_5Model(DecoderModel):
    def __init__(self, model_config: ModelConfig[Spark2_5Config]):
        super().__init__(model_config)
        config = model_config.pretrained_config

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        self.layers = nn.ModuleList(
            [
                Spark2_5DecoderLayer(model_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    @inference_mode_unless_compiling
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                **kwargs,
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        return hidden_states


@register_auto_model("Spark2_5ForCausalLM")
class Spark2_5ForCausalLM(DecoderModelForCausalLM[Spark2_5Model, Spark2_5Config]):
    def __init__(self, model_config: ModelConfig[Spark2_5Config]):
        super().__init__(
            Spark2_5Model(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config: Any = None) -> Literal["V2"]:
        """Prefer KV cache manager V2 for the hybrid SWA layout.

        V2 sizes the sliding-window and full-attention pools independently
        instead of statically dividing memory between them.
        """
        return "V2"

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional["SparkHfWeightMapper"] = None,
        params_map: Optional[Dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ):
        if weight_mapper is None:
            weight_mapper = SparkHfWeightMapper()
            weight_mapper.init_model_and_config(self, self.model_config)
        weights = weight_mapper.preprocess_weights(weights)

        super().load_weights(
            weights=weights,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )


@register_mapper("HF", "Spark2_5ForCausalLM")
class SparkHfWeightMapper(HfWeightMapper):
    """Maps Spark HF checkpoint tensor names onto TRT-LLM module names.

    The Spark checkpoint fuses QKV into a single ``self_attn.q_k_v_proj``
    tensor (q|k|v row order), names the embedding ``model.embedding`` and the
    output projection ``self_attn.out_proj``. The fused-QKV loader expects
    separate q/k/v entries, so the fused tensor is split here; the rest are
    renames. ``self_attn.g_proj`` and the MLP/norm names already match.
    """

    def preprocess_weights(self, weights: dict) -> dict:
        weights = self.rename_by_params_map(
            {
                r"model\.embedding\.": r"model.embed_tokens.",
                r"(.*)self_attn\.out_proj(.*)": r"\1self_attn.o_proj\2",
            },
            weights,
        )

        qkv_keys = [k for k in weights if ".self_attn.q_k_v_proj." in k]
        if qkv_keys:
            config = self._config.pretrained_config
            q_dim = config.num_attention_heads * config.head_dim
            kv_dim = config.num_key_value_heads * config.head_dim
            for key in qkv_keys:
                tensor = weights[key]
                del weights[key]
                prefix, suffix = key.split("q_k_v_proj")
                q, k, v = torch.split(tensor, [q_dim, kv_dim, kv_dim], dim=0)
                weights[f"{prefix}q_proj{suffix}"] = q
                weights[f"{prefix}k_proj{suffix}"] = k
                weights[f"{prefix}v_proj{suffix}"] = v

        return weights
