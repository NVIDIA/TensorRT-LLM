# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import re
from typing import Dict, List, Optional

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.utils import ActivationType, relu2

from ..attention_backend import AttentionMetadata
from ..distributed import AllReduce
from ..model_config import ModelConfig
from ..modules.attention import Attention
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import MoEWeightLoadingMode, create_moe
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mamba.mamba2_mixer import Mamba2Mixer
from ..modules.mlp import MLP
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType
from .modeling_deepseekv3 import DeepseekV3MTPHead
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, register_auto_model


class NemotronHConfig(PretrainedConfig):
    model_type = "nemotron_h"


class MLPLayer(MLP):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
    ):
        config = model_config.pretrained_config
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[layer_idx]
        else:
            intermediate_size = config.intermediate_size

        super().__init__(hidden_size=config.hidden_size,
                         intermediate_size=intermediate_size,
                         bias=False,
                         activation=relu2,
                         dtype=config.torch_dtype,
                         config=model_config)
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(hidden_states)


class TransformerLayer(Attention):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
        reduce_output: bool = False,
    ):
        config = model_config.pretrained_config

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=None,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            reduce_output=reduce_output,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(position_ids=None,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata)


# Ref code: https://huggingface.co/nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024/blob/main/modeling_nemotron_h.py#L818
class NemotronHMOE(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()

        # Import here to avoid circular dependency.
        from .modeling_deepseekv3 import DeepseekV3Gate

        self.activation_type = ActivationType.Relu2
        self.reduce_results = False

        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.layer_idx = layer_idx
        self.moe_intermediate_size = config.moe_intermediate_size[0] \
            if isinstance(config.moe_intermediate_size, list) else config.moe_intermediate_size
        self.use_latent_moe: bool = getattr(config, "moe_latent_size",
                                            None) is not None
        self.moe_hidden_size: int = config.moe_latent_size if self.use_latent_moe else config.hidden_size
        self.mlp_bias = config.mlp_bias if hasattr(config,
                                                   'mlp_bias') else False
        self.moe_n_group = config.n_group
        self.num_experts = config.n_routed_experts
        self.hidden_size = config.hidden_size
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.routed_scaling_factor = config.routed_scaling_factor
        self.mapping = model_config.mapping

        # Setup shared expert MLP.
        if config.n_shared_experts is None or config.n_shared_experts == 0:
            self.shared_experts = None
        else:
            shared_expert_intermediate_size = (
                config.moe_shared_expert_intermediate_size *
                config.n_shared_experts)

            self.shared_experts = MLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                bias=self.mlp_bias,
                activation=relu2,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=self.layer_idx,
                reduce_output=False,
                overridden_tp_size=1
                if model_config.mapping.enable_attention_dp else None)
        # Setup MoE gate.
        self.gate = DeepseekV3Gate(
            self.hidden_size,
            self.num_experts,
            top_k=self.top_k,
            n_group=self.moe_n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            dtype=config.torch_dtype,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend=model_config.moe_backend)

        # Setup MoE experts.
        self.experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=self.moe_intermediate_size,
            aux_stream_dict=aux_stream_dict,
            dtype=config.torch_dtype,
            reduce_results=self.reduce_results,
            model_config=model_config,
            layer_idx=self.layer_idx,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
            bias=self.mlp_bias,
            activation_type=self.activation_type,
        )

        if not model_config.mapping.enable_attention_dp:
            # AllReduce for combining shared and routed expert outputs in multi-GPU settings.
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
            )
        else:
            self.allreduce = None

        # Setup latent projection layers.
        # These layers should NOT be TP-sharded to ensure MoE receives
        # full latent representation. They are replicated across all GPUs.
        if self.use_latent_moe:
            self.fc1_latent_proj = Linear(
                in_features=self.hidden_size,
                out_features=self.moe_hidden_size,
                bias=self.mlp_bias,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
            )
            self.fc2_latent_proj = Linear(
                in_features=self.moe_hidden_size,
                out_features=self.hidden_size,
                bias=self.mlp_bias,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

        self.aux_stream_shared = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.MoeShared]
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        def _compute_shared_output():
            if self.shared_experts is not None:
                shared_expert_output = self.shared_experts(hidden_states)
            else:
                shared_expert_output = 0
            return shared_expert_output

        def _compute_routed_output():
            router_logits = self.gate(hidden_states)

            routed_hidden_states = hidden_states
            if self.use_latent_moe:
                routed_hidden_states = self.fc1_latent_proj(
                    routed_hidden_states)

            final_hidden_states = self.experts(
                routed_hidden_states,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False)

            if self.use_latent_moe:
                final_hidden_states = self.fc2_latent_proj(final_hidden_states)

            return final_hidden_states

        routed_output, shared_output = maybe_execute_in_parallel(
            _compute_routed_output, _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared], self.aux_stream_shared)

        final_hidden_states = shared_output + routed_output

        # Perform all-reduce after combining outputs for multi-GPU support.
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            final_hidden_states = self.allreduce(final_hidden_states)

        return final_hidden_states.view(orig_shape)


class NemotronHLayer(DecoderLayer):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
        # M -> MambaLayer
        # - -> MLPLayer
        # * -> TransformerLayer
        layer_type: str,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__()

        config = model_config.pretrained_config

        self.layer_idx = layer_idx
        self.layer_type = layer_type

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        if layer_type == "M":
            self.mixer = Mamba2Mixer(d_model=config.hidden_size,
                                     d_state=config.ssm_state_size,
                                     d_conv=config.conv_kernel,
                                     nheads=config.mamba_num_heads,
                                     n_groups=config.n_groups,
                                     head_dim=config.mamba_head_dim,
                                     chunk_size=config.chunk_size,
                                     layer_idx=layer_idx,
                                     rms_norm_eps=config.rms_norm_eps,
                                     dtype=config.torch_dtype,
                                     config=model_config)
        elif layer_type == "-":
            self.mixer = MLPLayer(model_config, layer_idx)
        elif layer_type == "*":
            self.mixer = TransformerLayer(
                model_config,
                layer_idx,
                reduce_output=not model_config.mapping.enable_attention_dp
                and model_config.mapping.tp_size > 1)
        elif layer_type == "E":
            self.mixer = NemotronHMOE(model_config,
                                      layer_idx=layer_idx,
                                      aux_stream_dict=aux_stream_dict)
        else:
            raise ValueError(f"{layer_type} is not supported")

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = self.mixer(hidden_states,
                                   attn_metadata,
                                   spec_metadata=spec_metadata,
                                   **kwargs)
        hidden_states = torch.add(hidden_states, residual)
        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, None)

        return hidden_states


class NemotronHModel(DecoderModel):

    def __init__(self, model_config: ModelConfig[NemotronHConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.aux_stream_dict = {
            # TODO: add attention stream.
            # AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared:
            aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap:
            aux_stream_list[1],
            AuxStreamType.MoeBalancer:
            aux_stream_list[2],
        }

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        else:
            self.embed_tokens = Embedding(
                config.vocab_size,
                config.hidden_size,
                dtype=config.torch_dtype,
                mapping=model_config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        # create layers
        layers = []
        for layer_idx, layer_type in enumerate(config.hybrid_override_pattern):
            layers.append(
                NemotronHLayer(model_config,
                               layer_idx,
                               layer_type,
                               aux_stream_dict=self.aux_stream_dict))
        self.layers = nn.ModuleList(layers)
        self.num_hidden_layers = config.num_hidden_layers

        # final norm
        self.norm_f = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        self.mamba_metadata: Optional[Mamba2Metadata] = None

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
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.mamba_metadata is None or self.mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
            self.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests,
                chunk_size=self.model_config.pretrained_config.chunk_size)
        self.mamba_metadata.prepare(attn_metadata)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        for layer in self.layers[:self.num_hidden_layers]:
            hidden_states = layer(position_ids,
                                  hidden_states,
                                  attn_metadata,
                                  spec_metadata=spec_metadata,
                                  mamba_metadata=self.mamba_metadata)

        hidden_states = self.norm_f(hidden_states)

        return hidden_states


@register_auto_model("NemotronHForCausalLM")
class NemotronHForCausalLM(SpecDecOneEngineForCausalLM[NemotronHModel,
                                                       NemotronHConfig]):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
    ):
        # rms_norm_eps might be named differently in the config.
        if hasattr(model_config.pretrained_config, "rms_norm_eps"):
            rms_epsilon = model_config.pretrained_config.rms_norm_eps
        elif hasattr(model_config.pretrained_config, "layer_norm_epsilon"):
            rms_epsilon = model_config.pretrained_config.layer_norm_epsilon
        else:
            raise ValueError("layer_norm_epsilon or rms_norm_eps is not set")
        model_config.pretrained_config.rms_norm_eps = rms_epsilon

        if not model_config.mapping.tp_size in [1, 2, 4, 8]:
            raise ValueError("TP has to be either 1, 2, 4 or 8")

        if model_config.quant_config.exclude_modules is not None:
            model_config.quant_config.exclude_modules = [
                re.sub(r'(model\.layers\.)?backbone', 'model', k)
                for k in model_config.quant_config.exclude_modules
            ]

        super().__init__(
            model=NemotronHModel(model_config),
            model_config=model_config,
        )
        self.model_nextn = 0
        if model_config.spec_config is not None and model_config.spec_config.spec_dec_mode.is_mtp_one_model(
        ):
            model_nextn = model_config.spec_config.num_nextn_predict_layers
            ckpt_nextn = self.config.num_nextn_predict_layers
            self.num_hidden_layers = self.config.num_hidden_layers
            assert ckpt_nextn > 0, "There are not MTP modules in the checkpoint."
            if ckpt_nextn == 1 and not model_config.spec_config.use_mtp_vanilla:
                pass
            else:
                # modify the QuantConfig to support duplicated mtp layers
                if model_config.quant_config.exclude_modules is not None:
                    extend_exclude_modules = []
                    for model_mtp_idx in range(
                            self.num_hidden_layers,
                            self.num_hidden_layers + model_nextn):
                        ckpt_mtp_idx = (model_mtp_idx - self.num_hidden_layers
                                        ) % ckpt_nextn + self.num_hidden_layers
                        model_prefix = f"model.layers.{model_mtp_idx}"
                        ckpt_prefix = f"model.layers.{ckpt_mtp_idx}"
                        for exclude_module in model_config.quant_config.exclude_modules:
                            if ckpt_prefix in exclude_module and model_prefix not in exclude_module:
                                extend_exclude_modules.append(
                                    exclude_module.replace(
                                        ckpt_prefix, model_prefix))
                    self.model_config.quant_config.exclude_modules.extend(
                        extend_exclude_modules)
            self.model.layers.extend(self.draft_model.mtp_layers)
            self.epilogue.extend(self.draft_model.mtp_layers)
            self.epilogue.append(self.spec_worker)

    def load_weights(self, weights: Dict, weight_mapper: BaseWeightMapper):
        new_weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(weights=new_weights, weight_mapper=weight_mapper)


class NemotronHMTPDecoderLayer(NemotronHLayer):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        has_start_projections: bool,
        has_end_norm: bool,
        layer_type: str,
    ) -> None:

        super().__init__(
            model_config=model_config,
            layer_idx=layer_idx,
            layer_type=layer_type,
            aux_stream_dict=aux_stream_dict,
        )
        self.model_nextn = 0
        if model_config.spec_config is not None and model_config.spec_config.spec_dec_mode.is_mtp_one_model(
        ):
            model_nextn = model_config.spec_config.num_nextn_predict_layers
            ckpt_nextn = self.config.num_nextn_predict_layers
            self.num_hidden_layers = self.config.num_hidden_layers
            assert ckpt_nextn > 0, "There is not MTP modules in the checkpoint."
            if ckpt_nextn == 1 and not model_config.spec_config.use_mtp_vanilla:
                pass
            else:
                # modify the QuantConfig to support duplicated mtp layers
                if model_config.quant_config.exclude_modules is not None:
                    extend_exclude_modules = []
                    for model_mtp_idx in range(
                            self.num_hidden_layers,
                            self.num_hidden_layers + model_nextn):
                        ckpt_mtp_idx = (model_mtp_idx - self.num_hidden_layers
                                        ) % ckpt_nextn + self.num_hidden_layers
                        model_prefix = f"model.layers.{model_mtp_idx}"
                        ckpt_prefix = f"model.layers.{ckpt_mtp_idx}"
                        for exclude_module in model_config.quant_config.exclude_modules:
                            if ckpt_prefix in exclude_module and model_prefix not in exclude_module:
                                extend_exclude_modules.append(
                                    exclude_module.replace(
                                        ckpt_prefix, model_prefix))
                    self.model_config.quant_config.exclude_modules.extend(
                        extend_exclude_modules)
            self.model.layers.extend(self.draft_model.mtp_layers)
            self.epilogue.extend(self.draft_model.mtp_layers)
            self.epilogue.append(self.spec_worker)

        config = model_config.pretrained_config
        self.model_config = model_config
        self.has_start_projections = has_start_projections
        self.has_end_norm = has_end_norm

        if has_start_projections:
            self.enorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )
            self.hnorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )

            if model_config.mapping.enable_attention_dp:
                self.eh_proj = Linear(
                    in_features=config.hidden_size * 2,
                    out_features=config.hidden_size,
                    bias=False,
                    dtype=config.torch_dtype,
                    quant_config=model_config.quant_config,
                    skip_create_weights_in_init=model_config.
                    skip_create_weights_in_init,
                )
            else:
                self.eh_proj = Linear(
                    in_features=config.hidden_size * 2,
                    out_features=config.hidden_size,
                    bias=False,
                    dtype=config.torch_dtype,
                    tensor_parallel_mode=TensorParallelMode.ROW,
                    mapping=model_config.mapping,
                    quant_config=model_config.quant_config,
                    reduce_output=True,
                    skip_create_weights_in_init=model_config.
                    skip_create_weights_in_init,
                )

        if has_end_norm:
            self.final_layernorm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
            )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        attn_metadata: Optional[AttentionMetadata] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        if self.has_start_projections:
            assert inputs_embeds is not None
            inputs_embeds_normed = self.enorm(inputs_embeds)
            previous_hidden_states_normed = self.hnorm(hidden_states)

            # Fuse via concatenation and linear projection
            fused = torch.cat(
                [inputs_embeds_normed, previous_hidden_states_normed], dim=-1)

            # Split fused hidden_states columnwise based on TP
            mapping = self.model_config.mapping
            if mapping.tp_size > 1 and not mapping.enable_attention_dp:
                fused = torch.chunk(fused, mapping.tp_size,
                                    dim=-1)[mapping.tp_rank]

            hidden_states = self.eh_proj(fused)
            residual = None  # Start fresh after fusion

        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
        )

        if self.has_end_norm:
            if residual is not None:
                hidden_states = hidden_states + residual
                residual = None

            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states, residual


class NemotronHMTP(nn.Module):
    """NemotronH MTP Layer - single MTP layer following DeepseekV3MTP pattern."""

    def __init__(self,
                 model_config: ModelConfig[NemotronHConfig],
                 layer_idx: int,
                 aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
                 is_separate_draft_engine: bool = False,
                 prefix: str = ""):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.config = config
        self.layer_idx = layer_idx

        # Pattern configuration
        self.pattern_str = config.mtp_hybrid_override_pattern
        self.pattern_len = len(self.pattern_str)
        assert self.pattern_len > 0

        # Build pattern-based layers
        self.layers = nn.ModuleDict()

        for i in range(self.pattern_len):
            step_rel_idx = i % self.pattern_len

            char = self.pattern_str[step_rel_idx]

            is_start_of_step = step_rel_idx == 0
            is_end_of_step = step_rel_idx == self.pattern_len - 1

            sublayer_quant_config = self._get_mtp_sublayer_quant_config(
                model_config, self.layer_idx)

            # Create a temporary model_config with the override quant_config
            sublayer_model_config = ModelConfig(
                pretrained_config=model_config.pretrained_config,
                mapping=model_config.mapping,
                quant_config=sublayer_quant_config,
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
            )

            self.layers[str(i)] = NemotronHMTPDecoderLayer(
                model_config=sublayer_model_config,
                layer_idx=self.layer_idx,
                aux_stream_dict=aux_stream_dict,
                has_start_projections=is_start_of_step,
                has_end_norm=is_end_of_step,
                layer_type=char,
            )

        # Add shared_head for MTP, following DeepseekV3MTP pattern
        self.shared_head = DeepseekV3MTPHead(model_config)

    def _get_mtp_sublayer_quant_config(
            self, model_config: ModelConfig[NemotronHConfig], layer_idx: int):
        """
        Get quantization config for MTP sublayer.
        The MTP layer in the nvfp4 checkpoint is unquantized. Because the TRTLLM
        moe_backend only supports fp8/fp4 quantization, we need to override
        the quant_config for the MTP layer.
        """
        from tensorrt_llm.models.modeling_utils import QuantConfig
        quant_config = model_config.quant_config
        # MTP layers are always unquantized, force quant_algo=None
        if quant_config is None:
            return None
        return QuantConfig(
            quant_algo=None,
            kv_cache_quant_algo=quant_config.kv_cache_quant_algo,
        )

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        inputs_embeds = embed_tokens(input_ids)
        residual = None
        for i in range(self.pattern_len):
            layer = self.layers[str(i)]
            hidden_states, residual = layer(
                inputs_embeds=inputs_embeds,
                positions=position_ids,
                hidden_states=hidden_states,
                residual=residual,
                attn_metadata=attn_metadata,
            )
        return hidden_states


AutoConfig.register(NemotronHConfig.model_type, NemotronHConfig)
