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
"""Engine-path model for the Qwen3.5/Qwen3.6 hybrid MoE family.

Registers the architecture, parses the config, and assembles the hybrid decoder
structure: Gated DeltaNet "linear attention" layers interleaved 3:1 with full
attention, each followed by a 256-expert MoE + shared expert. It is built on the
RecurrentGemma hybrid template, which interleaves recurrent and attention blocks
in a single engine.

The Gated DeltaNet recurrence runs through the ``GatedDeltaNet`` TensorRT plugin
(``functional.gated_delta_rule``), carrying its per-(request, v-head) state on
the paged RNN-state path alongside the depthwise conv state, exactly like the
Mamba / RecurrentGemma recurrent layers. The full-attention layers reuse the
engine ``Attention`` module with the ``attn_output_gate`` sigmoid output gate.

Not implemented in this initial engine-path drop:

* Interleaved MRoPE (``mrope_section``) for multimodal positions. Text-only
  positions collapse to standard RoPE, so text generation is unaffected.
* The MTP (multi-token-prediction) head.
* The Qwen3-VL vision tower.
"""

from collections import OrderedDict
from typing import Optional

import tensorrt as trt

from ..._common import default_net
from ..._utils import pad_vocab_size, str_dtype_to_trt
from ...functional import (
    ACT2FN,
    LayerNormType,
    Tensor,
    arange,
    concat,
    exp,
    expand,
    gated_delta_rule,
    gather_last_token_logits,
    repeat_interleave,
    shape,
    sigmoid,
    softplus,
    split,
    unsqueeze,
    view,
)
from ...layers import (
    Attention,
    AttentionMaskType,
    AttentionParams,
    ColumnLinear,
    Embedding,
    KeyValueCacheParams,
    RmsNorm,
    SharedMoE,
)
from ...layers.ssm import MambaConv1d
from ...mapping import Mapping
from ...module import Module, ModuleList
from ...parameter import Parameter
from ..generation_mixin import GenerationMixin
from ..modeling_utils import PretrainedModel, QuantConfig
from .config import FULL_ATTENTION, LINEAR_ATTENTION, Qwen3NextConfig


class Qwen3NextGatedDeltaNet(Module):
    """Gated DeltaNet ("linear attention") mixer.

    Construction wires the checkpoint parameter layout:

      * ``in_proj_qkv``: hidden -> [Q | K | V] = key_dim*2 + value_dim
      * ``in_proj_z``  : hidden -> value_dim (output gate)
      * ``in_proj_a``  : hidden -> num_v_heads (decay logits)
      * ``in_proj_b``  : hidden -> num_v_heads (beta logits)
      * ``conv1d``     : depthwise causal conv over [Q|K|V], kernel 4
      * ``A_log``, ``dt_bias``: per value-head
      * ``norm``       : gated RMSNorm over head_v_dim
      * ``out_proj``   : value_dim -> hidden

    The recurrence (chunked prefill + single-step decode against a
    ``[B, num_v_heads, head_k, head_v]`` recurrent state and a
    ``[B, conv_dim, kernel-1]`` conv state) runs through the ``GatedDeltaNet``
    plugin via ``functional.gated_delta_rule``.
    """

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.dtype = config.dtype

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel = config.linear_conv_kernel_dim

        # Parameter modules mirror the HF split-projection checkpoint layout.
        self.in_proj_qkv = ColumnLinear(
            config.hidden_size,
            self.conv_dim,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=False,
        )
        self.in_proj_z = ColumnLinear(
            config.hidden_size,
            self.value_dim,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=False,
        )
        self.in_proj_a = ColumnLinear(
            config.hidden_size,
            self.num_v_heads,
            bias=False,
            dtype=config.dtype,
            gather_output=False,
        )
        self.in_proj_b = ColumnLinear(
            config.hidden_size,
            self.num_v_heads,
            bias=False,
            dtype=config.dtype,
            gather_output=False,
        )
        self.conv1d = MambaConv1d(
            self.conv_dim, d_conv=self.conv_kernel, dtype=config.dtype, apply_silu=True
        )
        # Per-(value-)head decay/timestep params (kept fp32 for the recurrence).
        self.A_log = Parameter(shape=(self.num_v_heads,), dtype="float32")
        self.dt_bias = Parameter(shape=(self.num_v_heads,), dtype="float32")
        self.norm = RmsNorm(
            normalized_shape=self.head_v_dim, eps=config.norm_epsilon, dtype=config.dtype
        )
        self.out_proj = ColumnLinear(
            self.value_dim,
            config.hidden_size,
            bias=False,
            dtype=config.dtype,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            gather_output=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        conv_state: Tensor,
        rnn_state: Tensor,
        host_request_types: Tensor,
        last_token_ids: Tensor,
        host_context_lengths=None,
        slot_mapping=None,
        conv_indices=None,
    ):
        """Gated DeltaNet forward.

        Returns (out, present_conv_state, present_rnn_state).
        """
        gqa = self.num_v_heads // self.num_k_heads

        # 1. projections
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [.., conv_dim]
        z = self.in_proj_z(hidden_states)  # [.., value_dim]
        a = self.in_proj_a(hidden_states)  # [.., num_v_heads]
        b = self.in_proj_b(hidden_states)  # [.., num_v_heads]

        # 2. depthwise causal conv over [Q|K|V] (+ SiLU), updates conv_state
        mixed_qkv, present_conv = self.conv1d(
            mixed_qkv,
            conv_state,
            host_request_types,
            last_token_ids,
            host_context_lengths,
            slot_mapping,
            conv_indices,
        )

        # 3. split into q,k,v and shape into heads; GQA-expand q,k to num_v_heads
        q, k, v = split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        remove_padding = default_net().plugin_config.remove_input_padding

        def _heads(x, n_heads, head_dim):
            # [.., n_heads*head_dim] -> [.., n_heads, head_dim] (packed or padded)
            if remove_padding:
                new_shape = concat([shape(x, 0), n_heads, head_dim])
            else:
                new_shape = concat([shape(x, 0), shape(x, 1), n_heads, head_dim])
            return view(x, new_shape)

        q = _heads(q, self.num_k_heads, self.head_k_dim)
        k = _heads(k, self.num_k_heads, self.head_k_dim)
        v = _heads(v, self.num_v_heads, self.head_v_dim)
        if gqa > 1:
            # repeat_interleave key heads to match value heads
            head_dim_axis = 1 if remove_padding else 2
            q = repeat_interleave(q, gqa, dim=head_dim_axis)
            k = repeat_interleave(k, gqa, dim=head_dim_axis)

        # 4. per-head gates: beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
        #    (softplus with PyTorch defaults beta=1.0, threshold=20.0, matching
        #    the HF reference F.softplus(a + dt_bias)).
        beta = sigmoid(b)
        g = (0.0 - exp(self.A_log.value.cast("float32"))) * softplus(
            a.cast("float32") + self.dt_bias.value, 1.0, 20.0
        )

        # 5. gated delta rule recurrence (updates rnn_state). The GatedDeltaNet
        #    plugin is fp32-only (state is always fp32; correctness-first path),
        #    so q/k/v/beta must be fp32 too (g is already fp32 above). The fp32
        #    state matches the recurrence's numerical requirements.
        q = q.cast("float32")
        k = k.cast("float32")
        v = v.cast("float32")
        beta = beta.cast("float32")
        y, present_rnn = gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            rnn_state,
            host_request_types,
            last_token_ids,
            num_v_heads=self.num_v_heads,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            chunk_size=64,
            use_qk_l2norm=True,
            dtype=self.config.mamba_ssm_dtype,
            host_context_lengths=host_context_lengths,
            slot_mapping=slot_mapping,
        )

        # 6. gated RMSNorm over head_v (norm-before-gate, SiLU(z)) then out_proj.
        #    The plugin emits y in fp32; bring it back to the model dtype for the
        #    gated norm + out_proj (which run in the model dtype).
        y = y.cast(self.dtype)
        y = self.norm(y)
        z = _heads(z, self.num_v_heads, self.head_v_dim)
        y = y * ACT2FN["silu"](z)
        if remove_padding:
            y = view(y, concat([shape(y, 0), self.value_dim]))
        else:
            y = view(y, concat([shape(y, 0), shape(y, 1), self.value_dim]))
        out = self.out_proj(y)
        return out, present_conv, present_rnn


class Qwen3NextAttention(Module):
    """Full-attention mixer for the 1-in-4 ``full_attention`` layers.

    Reuses the engine ``Attention`` module for QKV / partial-RoPE / GQA /
    qk_layernorm, plus its built-in ``attn_output_gate`` hook: a sigmoid gate
    applied to the attention output before ``o_proj``. In the HF checkpoint this
    gate is fused into a doubled ``q_proj`` (``[2 * num_heads * head_dim,
    hidden]``; per head the first ``head_dim`` rows are real Q, the next
    ``head_dim`` are the gate). The converter de-interleaves those halves:
    real-Q rows feed the fused ``attention.qkv`` Q-section, gate rows feed
    ``attention.gate``.

    NOTE: interleaved MRoPE (``mrope_section``) is not wired here. For text-only
    positions MRoPE collapses to standard RoPE, so this layer matches HF for
    text; multimodal (image) position parity additionally requires MRoPE.
    """

    def __init__(self, config: Qwen3NextConfig, layer_idx: int, attn_layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_output_gate = config.attn_output_gate

        # attn_layer_idx is this layer's index into the KV cache (i.e. the count
        # of full-attention layers at or before it), NOT the global layer index.
        self.attention = Attention(
            local_layer_idx=attn_layer_idx,
            hidden_size=config.hidden_size,
            attention_head_size=config.head_size,
            num_attention_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            dtype=config.dtype,
            attention_mask_type=AttentionMaskType.causal,
            bias=config.attn_bias,
            position_embedding_type=config.position_embedding_type,
            rotary_embedding_base=config.rotary_base,
            rotary_embedding_scaling=config.rotary_scaling,
            rotary_embedding_percentage=config.partial_rotary_factor,
            tp_rank=config.mapping.tp_rank,
            tp_group=config.mapping.tp_group,
            tp_size=config.mapping.tp_size,
            quant_mode=config.quant_mode,
            dense_bias=False,
            qk_layernorm=True,
            layernorm_type=LayerNormType.RmsNorm,
            attn_output_gate=config.attn_output_gate,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
    ):
        """Full-attention forward.

        Returns ``context`` (or ``(context, present_kv)`` when ``use_cache``).
        The ``attn_output_gate`` is applied inside the engine ``Attention``
        module (see its ``forward``), so the FP8 output-quant wiring and the
        single TP all-reduce on ``o_proj`` remain intact.
        """
        return self.attention(
            hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
        )


def _build_mlp(config: Qwen3NextConfig):
    """Per-layer MoE (256 experts) + shared expert, matching the checkpoint."""
    mlp_kwargs = {
        "moe_config": config.moe,
        "mapping": config.mapping,
        "use_shared_gate": True,
        "use_side_stream": True,
    }
    # The shared-expert intermediate size is carried on the MoeConfig.
    config.moe.shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size
    return SharedMoE(
        hidden_size=config.hidden_size,
        ffn_hidden_size=config.moe_intermediate_size,
        hidden_act=config.hidden_act,
        dtype=config.dtype,
        bias=False,
        tp_group=config.mapping.tp_group,
        tp_size=config.mapping.tp_size,
        quant_mode=config.quant_mode,
        **mlp_kwargs,
    )


class Qwen3NextDecoderLayer(Module):
    """One hybrid decoder layer: (linear|full) mixer + MoE, with residuals."""

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.is_linear = config.is_linear_attention_layer(layer_idx)

        self.input_layernorm = RmsNorm(
            normalized_shape=config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype
        )
        if self.is_linear:
            self.linear_attn = Qwen3NextGatedDeltaNet(config, layer_idx)
        else:
            # KV-cache slot = number of full-attention layers at or before this
            # one, minus one (RecurrentGemma pattern for hybrid models).
            attn_kv_idx = (
                sum(1 for i in range(layer_idx + 1) if config.get_layer_type(i) == FULL_ATTENTION)
                - 1
            )
            self.self_attn = Qwen3NextAttention(config, layer_idx, attn_kv_idx)
        self.post_attention_layernorm = RmsNorm(
            normalized_shape=config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype
        )
        self.mlp = _build_mlp(config)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask=None,
        use_cache=False,
        kv_cache_params=None,
        attention_params=None,
        conv_state=None,
        rnn_state=None,
        host_request_types=None,
        last_token_ids=None,
        host_context_lengths=None,
        slot_mapping=None,
        conv_indices=None,
    ):
        """Returns (hidden, present_kv|None, present_conv|None, present_rnn|None)."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        present_kv = present_conv = present_rnn = None
        if self.is_linear:
            hidden_states, present_conv, present_rnn = self.linear_attn(
                hidden_states,
                conv_state=conv_state,
                rnn_state=rnn_state,
                host_request_types=host_request_types,
                last_token_ids=last_token_ids,
                host_context_lengths=host_context_lengths,
                slot_mapping=slot_mapping,
                conv_indices=conv_indices,
            )
        else:
            attn_out = self.self_attn(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=kv_cache_params,
                attention_params=attention_params,
            )
            if use_cache:
                hidden_states, present_kv = attn_out
            else:
                hidden_states = attn_out

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_kv, present_conv, present_rnn


class Qwen3NextModel(Module):
    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        self.config = config
        self.mapping = config.mapping
        if self.mapping.is_first_pp_rank():
            self.vocab_embedding = Embedding(
                config.vocab_size, config.hidden_size, dtype=config.dtype
            )
        layers_range = config.mapping.pp_layers(config.num_hidden_layers)
        self.layers = ModuleList([Qwen3NextDecoderLayer(config, idx) for idx in layers_range])
        if self.mapping.is_last_pp_rank():
            self.norm = RmsNorm(
                normalized_shape=config.hidden_size, eps=config.norm_epsilon, dtype=config.dtype
            )

    def forward(
        self,
        input_ids: Tensor,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        conv_states=None,
        rnn_states=None,
        host_request_types=None,
        last_token_ids=None,
        host_context_lengths=None,
        slot_mapping=None,
    ):
        hidden_states = self.vocab_embedding(input_ids)

        # conv_indices (only needed by the OOTB conv path, i.e. no mamba_conv1d
        # plugin); mirrors RecurrentGemma.
        conv_indices = None
        if not default_net().plugin_config.mamba_conv1d_plugin:
            batch_size = shape(input_ids, 0)
            dim = self.config.linear_num_value_heads * self.config.linear_value_head_dim
            d_conv = self.config.linear_conv_kernel_dim
            indices = expand(
                unsqueeze(arange(0, d_conv - 1, dtype="int32"), 0), concat([batch_size, d_conv - 1])
            )
            offsets = expand(unsqueeze(last_token_ids, 1), concat([batch_size, d_conv - 1]))
            indices = unsqueeze(indices + offsets, 1)
            conv_indices = expand(indices, concat([batch_size, dim, d_conv - 1]))

        present_kvs, present_convs, present_rnns = [], [], []
        for layer, past_kv, past_conv, past_rnn in zip(
            self.layers, kv_cache_params.past_key_value, conv_states, rnn_states
        ):
            hidden_states, present_kv, present_conv, present_rnn = layer(
                hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                kv_cache_params=KeyValueCacheParams(
                    past_key_value=[past_kv],
                    host_past_key_value_lengths=kv_cache_params.host_past_key_value_lengths,
                    host_max_attention_window_sizes=kv_cache_params.host_max_attention_window_sizes,
                    host_sink_token_length=kv_cache_params.host_sink_token_length,
                    kv_cache_block_offsets=kv_cache_params.kv_cache_block_offsets,
                    host_kv_cache_block_offsets=kv_cache_params.host_kv_cache_block_offsets,
                    host_kv_cache_pool_pointers=kv_cache_params.host_kv_cache_pool_pointers,
                    host_kv_cache_pool_mapping=kv_cache_params.host_kv_cache_pool_mapping,
                    cache_indirection=kv_cache_params.cache_indirection,
                ),
                attention_params=attention_params,
                conv_state=past_conv,
                rnn_state=past_rnn,
                host_request_types=host_request_types,
                last_token_ids=last_token_ids,
                host_context_lengths=host_context_lengths,
                slot_mapping=slot_mapping,
                conv_indices=conv_indices,
            )
            present_kvs.append(present_kv)
            present_convs.append(present_conv)
            present_rnns.append(present_rnn)

        hidden_states = self.norm(hidden_states)
        return hidden_states, tuple(present_kvs), tuple(present_convs), tuple(present_rnns)


class Qwen3NextForCausalLM(PretrainedModel):
    """Engine-path ``ForCausalLM`` for Qwen3.5/Qwen3.6 hybrid MoE checkpoints.

    Registers the architecture, constructs the full module tree (hybrid
    GatedDeltaNet + full-attention decoder, MoE), and wires the recurrent
    conv/RNN-state inputs alongside the KV cache so the engine builds and runs
    the text-generation path. The MTP head and vision tower are out of scope
    here (see module docstring).
    """

    config_class = Qwen3NextConfig

    def __init__(self, config: Qwen3NextConfig):
        self.check_config(config)
        super().__init__(config)
        dtype = config.dtype
        self.dtype = str_dtype_to_trt(dtype) if isinstance(dtype, str) else dtype
        self.quant_mode = config.quant_mode
        self.mapping = config.mapping
        self.config = config
        self.gather_context_logits = False

        # Per-layer temporal types (used by prepare_inputs / the forward loop to
        # split KV-cache vs recurrent-state layers).
        self.layer_types = [config.get_layer_type(i) for i in range(config.num_hidden_layers)]

        # Constant attention params shared by the full-attention layers.
        Attention.create_attention_const_params(self, config)
        self.position_embedding_type = config.position_embedding_type

        logits_dtype = config.logits_dtype
        self._logits_dtype = (
            str_dtype_to_trt(logits_dtype) if isinstance(logits_dtype, str) else logits_dtype
        )

        self.transformer = Qwen3NextModel(config)
        vocab_size_padded = pad_vocab_size(config.vocab_size, config.mapping.tp_size)
        if config.mapping.is_last_pp_rank():
            self.lm_head = ColumnLinear(
                config.hidden_size,
                vocab_size_padded,
                bias=False,
                dtype=dtype,
                tp_group=config.mapping.tp_group,
                tp_size=config.mapping.tp_size,
                gather_output=True,
            )
        else:
            self.lm_head = None

    def check_config(self, config):
        config.set_if_not_exist("linear_num_key_heads", 16)
        config.set_if_not_exist("linear_num_value_heads", 32)
        config.set_if_not_exist("linear_key_head_dim", 128)
        config.set_if_not_exist("linear_value_head_dim", 128)
        config.set_if_not_exist("linear_conv_kernel_dim", 4)
        config.set_if_not_exist("mamba_ssm_dtype", "float32")

    @classmethod
    def from_hugging_face(
        cls,
        hf_model_or_dir,
        dtype: str = "auto",
        mapping: Optional[Mapping] = None,
        quant_config: Optional[QuantConfig] = None,
        **kwargs,
    ):
        """Build the TRT-LLM model + load weights from an HF checkpoint dir.

        Parses the config (incl. the modelopt mixed-precision
        ``quantization_config`` -> per-layer ``LayerQuantConfig``), constructs
        the module tree (which ``quantize``-s the FP8/NVFP4 layers), then loads
        and remaps the HF safetensors via the unified ``ModelWeightsLoader``
        (see ``convert.py``). MTP + vision are not built here and are out of
        scope for the text-only path.
        """
        from .convert import load_weights_from_hf_model

        config = Qwen3NextConfig.from_hugging_face(
            hf_model_or_dir, dtype=dtype, mapping=mapping, quant_config=quant_config, **kwargs
        )
        model = cls(config)
        load_weights_from_hf_model(hf_model_or_dir, model)
        return model

    def forward(
        self,
        input_ids: Tensor,
        position_ids=None,
        use_cache=False,
        attention_mask=None,
        kv_cache_params=None,
        attention_params=None,
        conv_states=None,
        rnn_states=None,
        host_request_types=None,
        last_token_ids=None,
        last_token_ids_for_logits=None,
        host_context_lengths=None,
        slot_mapping=None,
    ):
        attention_params = Attention.fill_attention_params(self, attention_params)

        hidden_states, present_kvs, present_convs, present_rnns = self.transformer(
            input_ids,
            use_cache=use_cache,
            attention_mask=attention_mask,
            kv_cache_params=kv_cache_params,
            attention_params=attention_params,
            conv_states=conv_states,
            rnn_states=rnn_states,
            host_request_types=host_request_types,
            last_token_ids=last_token_ids,
            host_context_lengths=host_context_lengths,
            slot_mapping=slot_mapping,
        )

        hidden_states = gather_last_token_logits(
            hidden_states,
            last_token_ids_for_logits,
            default_net().plugin_config.remove_input_padding,
        )
        lm_logits = self.lm_head(hidden_states)
        lm_logits.mark_output("logits", self._logits_dtype)

        if not default_net().plugin_config.paged_kv_cache:
            for i, present_kv in enumerate(present_kvs):
                if present_kv is not None:
                    present_kv.mark_output(f"present_key_value_{i}", self.dtype)
        if not default_net().plugin_config.paged_state:
            for i, present_conv in enumerate(present_convs):
                if present_conv is not None:
                    present_conv.mark_output(f"present_conv_state_{i}", self.dtype)
            for i, present_rnn in enumerate(present_rnns):
                if present_rnn is not None:
                    present_rnn.mark_output(f"present_rnn_state_{i}", str_dtype_to_trt("float32"))
        return (lm_logits, present_kvs, present_convs, present_rnns)

    def prepare_recurrent_inputs(self, max_batch_size, num_profiles, mapping):
        """Build per-GDN-layer conv + recurrent (matrix) state inputs.

        Non-GDN layers get None placeholders so the per-layer lists align with
        self.layers.
        """
        paged = default_net().plugin_config.paged_state
        default_range = GenerationMixin.default_range
        batch_range = [default_range(max_batch_size)] * num_profiles

        cfg = self.config
        conv_dim = (
            cfg.linear_key_head_dim * cfg.linear_num_key_heads * 2
            + cfg.linear_value_head_dim * cfg.linear_num_value_heads
        ) // mapping.tp_size
        n_v_heads = cfg.linear_num_value_heads // mapping.tp_size
        d_conv = cfg.linear_conv_kernel_dim

        conv_state_dim_range = OrderedDict(
            [
                ("batch_size", batch_range),
                ("kernel_size", [d_conv - 1] * num_profiles),
                ("conv_dim", [conv_dim] * num_profiles),
            ]
        )
        # GDN recurrent state is the per-v-head matrix [head_k, head_v].
        rnn_state_dim_range = OrderedDict(
            [
                ("batch_size", batch_range),
                ("num_v_heads", [n_v_heads] * num_profiles),
                ("head_k_dim", [cfg.linear_key_head_dim] * num_profiles),
                ("head_v_dim", [cfg.linear_value_head_dim] * num_profiles),
            ]
        )
        one_dim_range = OrderedDict([("buffer_count", [1] * num_profiles)])

        conv_states, rnn_states = [], []
        for i in range(cfg.num_hidden_layers):
            if self.layer_types[i] != LINEAR_ATTENTION:
                conv_states.append(None)
                rnn_states.append(None)
                continue
            if paged:
                conv_state = Tensor(
                    name=f"conv_state_ptr_{i}",
                    dtype=str_dtype_to_trt("int64"),
                    shape=[1],
                    dim_range=one_dim_range,
                )
                rnn_state = Tensor(
                    name=f"rnn_state_ptr_{i}",
                    dtype=str_dtype_to_trt("int64"),
                    shape=[1],
                    dim_range=one_dim_range,
                )
            else:
                conv_state = Tensor(
                    name=f"past_conv_state_{i}",
                    dtype=self.dtype,
                    shape=[-1, d_conv - 1, conv_dim],
                    dim_range=conv_state_dim_range,
                )
                rnn_state = Tensor(
                    name=f"past_rnn_state_{i}",
                    dtype=str_dtype_to_trt("float32"),
                    shape=[-1, n_v_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim],
                    dim_range=rnn_state_dim_range,
                )
            conv_states.append(conv_state)
            rnn_states.append(rnn_state)

        slot_mapping = None
        if paged:
            slot_mapping = Tensor(
                name="slot_mapping",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("batch_size", batch_range)]),
            )
        return {
            "conv_states": conv_states,
            "rnn_states": rnn_states,
            "slot_mapping": slot_mapping,
        }

    def prepare_inputs(
        self,
        max_batch_size,
        max_input_len,
        max_seq_len,
        max_num_tokens,
        use_cache,
        max_beam_width: int = 1,
        opt_num_tokens: int = None,
        opt_batch_size: int = 0,
        prompt_embedding_table_size: int = 0,
        max_draft_len: int = 0,
        gather_context_logits: bool = False,
        lora_target_modules=None,
        speculative_decoding_draft_tokens_external: bool = False,
    ):
        """Prepare the hybrid input tensors.

        Standard LM + attention inputs for the full-attention layers, plus
        conv/recurrent-state inputs for the GDN layers. Mirrors
        RecurrentGemmaForCausalLM.prepare_inputs.
        """
        assert not speculative_decoding_draft_tokens_external, (
            "Qwen3-Next does not support external draft-token spec decoding."
        )
        assert max_beam_width == 1, "Qwen3-Next does not support beam search."

        from ..modeling_utils import get_kv_cache_type_from_legacy

        remove_input_padding = default_net().plugin_config.remove_input_padding
        use_gpt_attention_plugin = default_net().plugin_config.gpt_attention_plugin
        use_gemm_plugin = default_net().plugin_config.gemm_plugin
        paged_kv_cache = default_net().plugin_config.paged_kv_cache
        tokens_per_block = default_net().plugin_config.tokens_per_block
        multiple_profiles = default_net().plugin_config.multiple_profiles
        streamingllm = default_net().plugin_config.streamingllm

        mapping = self.config.mapping
        kv_cache_type = get_kv_cache_type_from_legacy(use_cache, paged_kv_cache)

        enable_ctx_gen_opt_profiles = GenerationMixin.has_ctx_gen_opt_profiles(
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            use_gemm_plugin=use_gemm_plugin,
            remove_input_padding=remove_input_padding,
            kv_cache_type=kv_cache_type,
        )
        num_profiles, ranges = GenerationMixin.get_profiles_ranges(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_num_tokens=max_num_tokens,
            max_draft_len=max_draft_len,
            opt_batch_size=opt_batch_size,
            opt_num_tokens=opt_num_tokens,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            multiple_profiles=multiple_profiles,
            kv_cache_type=kv_cache_type,
        )

        if remove_input_padding:
            input_ids = Tensor(
                name="input_ids",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict(
                    [
                        ("num_tokens", ranges["num_tokens_range"]),
                    ]
                ),
            )
            position_ids = Tensor(
                name="position_ids",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict(
                    [
                        ("position_ids_num_tokens_range", ranges["num_tokens_range"]),
                    ]
                ),
            )
        else:
            input_ids = Tensor(
                name="input_ids",
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict(
                    [
                        ("batch_size_beam_width", ranges["bb_range"]),
                        ("input_len", ranges["inlen_range"]),
                    ]
                ),
            )
            position_ids = Tensor(
                name="position_ids",
                dtype=trt.int32,
                shape=[-1, -1],
                dim_range=OrderedDict(
                    [
                        ("batch_size_beam_width", ranges["bb_range"]),
                        ("position_ids_inlen_range", ranges["position_ids_inlen_range"]),
                    ]
                ),
            )

        # KV cache only for the full-attention layers.
        attn_layer_idx = [
            i for i in range(self.config.num_hidden_layers) if self.layer_types[i] == FULL_ATTENTION
        ]
        attention_inputs = self.prepare_attention_inputs(
            max_batch_size=max_batch_size,
            max_beam_width=max_beam_width,
            max_input_len=max_input_len,
            max_seq_len=max_seq_len,
            num_kv_heads=self.config.num_key_value_heads,
            head_size=self.config.head_size,
            num_layers=self.config.num_hidden_layers,
            kv_dtype=str_dtype_to_trt(self.config.kv_dtype),
            num_profiles=num_profiles,
            enable_ctx_gen_opt_profiles=enable_ctx_gen_opt_profiles,
            remove_input_padding=remove_input_padding,
            use_gpt_attention_plugin=use_gpt_attention_plugin,
            kv_cache_type=kv_cache_type,
            tokens_per_block=tokens_per_block,
            mapping=mapping,
            streamingllm=streamingllm,
            attn_layer_idx=attn_layer_idx,
        )

        recurrent_inputs = self.prepare_recurrent_inputs(
            max_batch_size=max_batch_size, num_profiles=num_profiles, mapping=mapping
        )

        if use_gpt_attention_plugin:
            host_request_types = attention_inputs["host_request_types"]
        else:
            host_request_types = Tensor(
                name="host_request_types",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("batch_size_beam_width", ranges["bb_range"])]),
            )

        last_token_ids = Tensor(
            name="last_token_ids",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size_last_token_ids", ranges["bbd_range"])]),
        )
        last_token_ids_for_logits = None
        if not gather_context_logits:
            last_token_ids_for_logits = last_token_ids

        if use_gpt_attention_plugin and remove_input_padding:
            host_context_lengths = attention_inputs["host_context_lengths"]
        elif remove_input_padding:
            host_context_lengths = Tensor(
                name="host_context_lengths",
                dtype=trt.int32,
                shape=[-1],
                dim_range=OrderedDict([("batch_size_beam_width", ranges["bb_range"])]),
            )
        else:
            host_context_lengths = None

        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "use_cache": True,
            "attention_mask": attention_inputs["attention_mask"],
            "kv_cache_params": KeyValueCacheParams(
                past_key_value=attention_inputs["past_key_value"],
                host_past_key_value_lengths=attention_inputs["host_past_key_value_lengths"],
                host_max_attention_window_sizes=attention_inputs["host_max_attention_window_sizes"],
                host_sink_token_length=attention_inputs["host_sink_token_length"],
                kv_cache_block_offsets=attention_inputs["kv_cache_block_offsets"],
                host_kv_cache_block_offsets=attention_inputs["host_kv_cache_block_offsets"],
                host_kv_cache_pool_pointers=attention_inputs["host_kv_cache_pool_pointers"],
                host_kv_cache_pool_mapping=attention_inputs["host_kv_cache_pool_mapping"],
                cache_indirection=attention_inputs["cache_indirection"],
            ),
            "attention_params": AttentionParams(
                sequence_length=attention_inputs["sequence_length"],
                context_lengths=attention_inputs["context_lengths"],
                host_context_lengths=attention_inputs["host_context_lengths"],
                max_context_length=max_input_len,
                host_request_types=attention_inputs["host_request_types"],
                host_runtime_perf_knobs=attention_inputs["host_runtime_perf_knobs"],
                host_context_progress=attention_inputs["host_context_progress"],
            ),
            "conv_states": recurrent_inputs["conv_states"],
            "rnn_states": recurrent_inputs["rnn_states"],
            "host_request_types": host_request_types,
            "last_token_ids": last_token_ids,
            "last_token_ids_for_logits": last_token_ids_for_logits,
            "host_context_lengths": host_context_lengths,
            "slot_mapping": recurrent_inputs["slot_mapping"],
        }
