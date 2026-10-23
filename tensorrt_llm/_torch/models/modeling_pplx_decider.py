# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from transformers import Qwen3NextConfig

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PredefinedAttentionMask
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.fla.chunk import chunk_gated_delta_rule
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mamba.causal_conv1d import causal_conv1d_fn
from ..modules.mamba.fuse_elementwise_ops import (
    extract_transpose_prefill_slice,
    fused_gdn_post_conv,
)
from ..modules.mamba.gdn_mixer import Qwen3NextGatedDeltaNet
from ..speculative import SpecMetadata
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .modeling_qwen3_next import (
    Qwen3NextFullAttentionDecoderLayer,
    Qwen3NextLinearDecoderLayer,
    Qwen3NextModel,
    _precompute_fused_norm_weights,
)
from .modeling_utils import DecoderModelForCausalLM, register_auto_model

PPLX_DECIDER_ARCHITECTURE = "PplxDeciderForDecision"
PPLX_DECIDER_NUM_LABELS = 255


class PplxDeciderGatedDeltaNet(Qwen3NextGatedDeltaNet):
    """Qwen3.5 GDN with context-only execution from a zero recurrent state."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        all_reduce_params: Optional[AllReduceParams] = None,
        decider_cu_seqlens_i32: Optional[torch.Tensor] = None,
        decider_cu_seqlens_i64: Optional[torch.Tensor] = None,
        **_: object,
    ) -> torch.Tensor:
        if decider_cu_seqlens_i32 is None or decider_cu_seqlens_i64 is None:
            raise ValueError("pplx-decider requires packed sequence boundaries")

        mixed_qkv, z, a, b = self._compute_tokenwise_inputs(hidden_states)
        mixed_qkv_t = extract_transpose_prefill_slice(
            mixed_qkv,
            mixed_qkv.shape[0],
            0,
            mixed_qkv.shape[1],
        )
        mixed_qkv_t = causal_conv1d_fn(
            mixed_qkv_t,
            self.conv1d.weight,
            self.conv1d.bias,
            query_start_loc=decider_cu_seqlens_i32,
            activation=self.activation,
        )
        query, key, value, log_gate, beta = fused_gdn_post_conv(
            mixed_qkv_t,
            None,
            a,
            b,
            self.A_log,
            self.dt_bias,
            self.num_k_heads_per_tp,
            self.head_k_dim,
            self.num_v_heads_per_tp,
            self.head_v_dim,
        )
        attn_out, _ = chunk_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=log_gate,
            beta=beta,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=decider_cu_seqlens_i64,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        return self._postprocess_gdn_output(attn_out, z, all_reduce_params)


class PplxDeciderLinearDecoderLayer(Qwen3NextLinearDecoderLayer):
    linear_attn_cls = PplxDeciderGatedDeltaNet


class PplxDeciderFullAttentionDecoderLayer(Qwen3NextFullAttentionDecoderLayer):
    attention_mask = PredefinedAttentionMask.FULL


class PplxDeciderModel(Qwen3NextModel):
    decoder_layer_types = {
        "full_attention": PplxDeciderFullAttentionDecoderLayer,
        "linear_attention": PplxDeciderLinearDecoderLayer,
    }

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        lora_params: Optional[dict] = None,
        **_: object,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        if attn_metadata.seq_lens_cuda is None:
            raise ValueError("pplx-decider requires sequence lengths")
        if attn_metadata.num_contexts != attn_metadata.seq_lens_cuda.shape[0]:
            raise ValueError("pplx-decider supports context-only execution")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        seq_lens = attn_metadata.seq_lens_cuda.to(torch.int32)
        cu_seqlens_i32 = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=seq_lens.device),
                seq_lens.cumsum(0, dtype=torch.int32),
            )
        )
        cu_seqlens_i64 = cu_seqlens_i32.to(torch.int64)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers[: self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
                decider_cu_seqlens_i32=cu_seqlens_i32,
                decider_cu_seqlens_i64=cu_seqlens_i64,
            )
        return hidden_states


@register_auto_model("PplxDeciderForDecision")
class PplxDeciderForDecision(DecoderModelForCausalLM[PplxDeciderModel, Qwen3NextConfig]):
    """Return raw float32 decision logits with shape ``[batch, 255]``."""

    def __init__(self, model_config: ModelConfig[Qwen3NextConfig]):
        nn.Module.__init__(self)
        self.model_config = model_config
        self.model = PplxDeciderModel(model_config)

        config = model_config.pretrained_config
        self.num_labels = config.num_labels
        self.readout = Linear(
            config.hidden_size,
            self.num_labels,
            bias=False,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
            reduce_output=False,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.prologue = []
        self.epilogue = [self.readout]
        self.preload_weight_modules = self.model.preload_weight_modules

    @classmethod
    def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> dict:
        return {"attn_backend": "FLASHINFER"}

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = self.model(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        end_indices = attn_metadata.seq_lens_cuda.cumsum(0).long() - 1
        return self.readout(hidden_states[end_indices]).float()

    def load_weights(
        self,
        weights: dict,
        weight_mapper: BaseWeightMapper,
        params_map: Optional[dict[str, str]] = None,
        allow_partial_loading: bool = False,
    ) -> None:
        mapped = weight_mapper.preprocess_weights(
            weights, allow_partial_loading=allow_partial_loading
        )
        if hasattr(weights, "clear"):
            weights.clear()
        super().load_weights(
            mapped,
            weight_mapper=weight_mapper,
            params_map=params_map,
            allow_partial_loading=allow_partial_loading,
        )

    def setup_aliases(self) -> None:
        for index, layer in enumerate(self.model.layers[: self.config.num_hidden_layers]):
            layer.next_layer_layernorm = (
                self.model.norm
                if index == self.config.num_hidden_layers - 1
                else self.model.layers[index + 1].input_layernorm
            )

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        _precompute_fused_norm_weights(self)
