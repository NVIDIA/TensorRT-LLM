# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from torch import nn
from transformers import AutoConfig, PretrainedConfig

from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.utils import ActivationType, relu2
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_helper import LoraConfig

from ..attention_backend import AttentionMetadata
from ..distributed import AllReduce, AllReduceFusionOp, AllReduceParams
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
from ..peft.lora.layer import LoraLayer, LoraModuleType
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType, Fp4QuantizedTensor
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
        reduce_output: bool = True,
    ):
        config = model_config.pretrained_config
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[layer_idx]
        else:
            intermediate_size = config.intermediate_size

        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            bias=False,
            activation=relu2,
            dtype=config.torch_dtype,
            config=model_config,
            reduce_output=reduce_output,
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(hidden_states, lora_params=lora_params)


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
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(position_ids=None,
                               hidden_states=hidden_states,
                               attn_metadata=attn_metadata,
                               lora_params=lora_params,
                               **kwargs)


# Ref code: https://huggingface.co/nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024/blob/main/modeling_nemotron_h.py#L818
class NemotronHMOE(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        reduce_output: bool = False,
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
        self.moe_intermediate_size = (config.moe_intermediate_size[0]
                                      if isinstance(
                                          config.moe_intermediate_size, list)
                                      else config.moe_intermediate_size)
        self.use_latent_moe: bool = getattr(config, "moe_latent_size",
                                            None) is not None
        self.moe_hidden_size: int = (config.moe_latent_size
                                     if self.use_latent_moe else
                                     config.hidden_size)
        self.mlp_bias = config.mlp_bias if hasattr(config,
                                                   "mlp_bias") else False
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
                if model_config.mapping.enable_attention_dp else None,
                # Use shared expert LoRA types (distinct dimensions from routed MoE experts)
                lora_up_module_type=LoraModuleType.SHARED_EXPERT_H_TO_4H,
                lora_down_module_type=LoraModuleType.SHARED_EXPERT_4H_TO_H,
            )
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
            moe_backend=model_config.moe_backend,
        )

        # For MIXED_PRECISION models, the global quant_config has quant_algo=MIXED_PRECISION
        # which maps to QuantMode(0) (no quant). This would cause the MoE backend to select
        # UnquantizedFusedMoEMethod and allocate BF16 weight buffers, causing a shape mismatch
        # when loading NVFP4/W4A8_NVFP4_FP8 quantized expert weights.
        # Look up the per-expert quant config from quant_config_dict and use it for create_moe.
        moe_model_config = model_config
        if model_config.quant_config_dict is not None:
            experts_prefix = f"model.layers.{layer_idx}.mixer.experts."
            for key, cfg in model_config.quant_config_dict.items():
                if key.startswith(experts_prefix):
                    moe_model_config = replace(model_config, quant_config=cfg)
                    break

        # Setup MoE experts.
        self.experts = create_moe(
            routing_method=self.gate.routing_method,
            num_experts=self.num_experts,
            hidden_size=self.moe_hidden_size,
            intermediate_size=self.moe_intermediate_size,
            aux_stream_dict=aux_stream_dict,
            dtype=config.torch_dtype,
            reduce_results=self.reduce_results,
            model_config=moe_model_config,
            layer_idx=self.layer_idx,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
            bias=self.mlp_bias,
            activation_type=self.activation_type,
        )

        if reduce_output:
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
            self.fc1_latent_lora = None
            if model_config.lora_config is not None:
                self.fc1_latent_lora = LoraLayer(
                    [LoraModuleType.MOE_LATENT_FC1], [self.moe_hidden_size])
            self.fc1_latent_proj = Linear(
                in_features=self.hidden_size,
                out_features=self.moe_hidden_size,
                bias=self.mlp_bias,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
                lora=self.fc1_latent_lora,
            )
            self.fc2_latent_lora = None
            if model_config.lora_config is not None:
                self.fc2_latent_lora = LoraLayer(
                    [LoraModuleType.MOE_LATENT_FC2], [self.hidden_size])
            self.fc2_latent_proj = Linear(
                in_features=self.moe_hidden_size,
                out_features=self.hidden_size,
                bias=self.mlp_bias,
                dtype=config.torch_dtype,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.
                skip_create_weights_in_init,
                lora=self.fc2_latent_lora,
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None
            self.fc1_latent_lora = None
            self.fc2_latent_lora = None

        self.aux_stream_shared = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {
            key: torch.cuda.Event()
            for key in [EventType.Main, EventType.MoeShared]
        }

    def forward(
        self,
        hidden_states: torch.Tensor
        | tuple[torch.Tensor | Fp4QuantizedTensor, torch.Tensor],
        attn_metadata: AttentionMetadata,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_hp = hidden_states
        else:
            hidden_states_hp = hidden_states

        assert hidden_states_hp.shape[-1] == self.hidden_dim
        orig_shape = hidden_states_hp.shape
        hidden_states_hp_2d = hidden_states_hp.view(-1, self.hidden_dim)
        # MTP sublayer may pass a corrected all_rank_num_tokens via kwargs,
        # since attn_metadata still holds the main model's token count.
        all_rank_num_tokens = kwargs.get('all_rank_num_tokens',
                                         attn_metadata.all_rank_num_tokens)

        def _compute_shared_output():
            if self.shared_experts is not None:
                shared_expert_output = self.shared_experts(
                    hidden_states_hp, lora_params=lora_params)
            else:
                shared_expert_output = 0
            return shared_expert_output

        def _compute_routed_output():
            # Gate uses high precision input for accurate routing decisions.
            router_logits = self.gate(hidden_states_hp_2d)

            if self.use_latent_moe:
                routed_hidden_states = self.fc1_latent_proj(
                    hidden_states_hp,
                    lora_params=lora_params,
                    layer_idx=self.layer_idx)
            else:
                routed_hidden_states = hidden_states

            final_hidden_states = self.experts(
                routed_hidden_states,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=False,
            )

            if self.use_latent_moe:
                final_hidden_states = self.fc2_latent_proj(
                    final_hidden_states,
                    lora_params=lora_params,
                    layer_idx=self.layer_idx)

            return final_hidden_states

        routed_output, shared_output = maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream_shared,
        )

        final_hidden_states = shared_output + routed_output

        # Perform all-reduce after combining outputs for multi-GPU support.
        if self.allreduce is not None:
            final_hidden_states = self.allreduce(
                final_hidden_states,
                all_reduce_params=kwargs.get('all_reduce_params'))

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
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        fuse_allreduce_norm: bool = False,
    ):
        super().__init__()

        config = model_config.pretrained_config

        self.layer_idx = layer_idx
        self.layer_type = layer_type

        quant_mode = (model_config.quant_config.quant_mode
                      if model_config.quant_config is not None else None)
        self.is_nvfp4 = quant_mode is not None and quant_mode.has_nvfp4()
        # For MIXED_PRECISION models, the global quant_mode is QuantMode(0). Check per-layer
        # quant_config_dict to see if this specific layer is NVFP4-quantized.
        if not self.is_nvfp4 and model_config.quant_config_dict is not None:
            layer_prefix = f"model.layers.{layer_idx}."
            for key, cfg in model_config.quant_config_dict.items():
                if key.startswith(layer_prefix) and cfg.quant_mode.has_nvfp4():
                    self.is_nvfp4 = True
                    break
        # The fused RMSNorm+NVFP4 CUDA kernel requires hidden_size to be
        # a supported tile size. Non-power-of-2 hidden sizes within tile
        # ranges may cause kernel hangs. Disable fused NVFP4 for such cases.
        # Supported tile sizes: 2048, 4096, 8192, 16384
        _SUPPORTED_NVFP4_HIDDEN_SIZES = {2048, 4096, 8192, 16384}
        if self.is_nvfp4 and config.hidden_size not in _SUPPORTED_NVFP4_HIDDEN_SIZES:
            logger.warning_once(
                f"Layer {layer_idx}: Disabling fused NVFP4 RMSNorm for hidden_size={config.hidden_size}. "
                f"Supported sizes: {_SUPPORTED_NVFP4_HIDDEN_SIZES}. Using non-fused path.",
                key=f"disable_nvfp4_rmsnorm_with_{config.hidden_size}",
            )
            self.is_nvfp4 = False
        # LoRA layers require regular bf16 tensors, not Fp4QuantizedTensor.
        # Disable fused RMSNorm+NVFP4 when LoRA is configured.
        if self.is_nvfp4 and model_config.lora_config is not None:
            self.is_nvfp4 = False

        # fuse_allreduce_norm is the model-level flag.  When enabled, ALL
        # layers defer mixer AllReduce to the next layer's pre_allreduce (or
        # the model's final_allreduce).  Only layers 1+ create a pre_allreduce
        # module; layer 0's input is already reduced from the embedding.
        self.fuse_allreduce_norm = fuse_allreduce_norm
        self.is_moe_layer = (layer_type == "E")

        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            # Enable fused NVFP4 quantization if possible.
            # It might be overridden in `_try_attach_nvfp4_scale` function.
            quantize_type="nvfp4" if self.is_nvfp4 else None,
            # Enable high precision output for MoE layer (only with NVFP4).
            # It might be overridden in `_try_attach_nvfp4_scale` function.
            return_hp_output=self.is_moe_layer and self.is_nvfp4,
        )

        if fuse_allreduce_norm and layer_idx > 0:
            self.pre_allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
            )

        # Mixer creation.  The fuse_allreduce_norm optimization is orthogonal
        # to AllReduce topology: Transformer/MoE gate it at forward time via
        # AllReduceParams; MLP/Mamba gate it at init time via reduce_output
        # (their base classes don't thread all_reduce_params through forward).
        has_tp_allreduce = (not model_config.mapping.enable_attention_dp
                            and model_config.mapping.tp_size > 1)

        if layer_type == "M":
            self.mixer = Mamba2Mixer(
                d_model=config.hidden_size,
                d_state=config.ssm_state_size,
                d_conv=config.conv_kernel,
                nheads=config.mamba_num_heads,
                n_groups=config.n_groups,
                head_dim=config.mamba_head_dim,
                chunk_size=config.chunk_size,
                layer_idx=layer_idx,
                rms_norm_eps=config.rms_norm_eps,
                dtype=config.torch_dtype,
                config=model_config,
            )
            if fuse_allreduce_norm:
                self.mixer.out_proj.reduce_output = False
        elif layer_type == "-":
            self.mixer = MLPLayer(
                model_config,
                layer_idx,
                reduce_output=not fuse_allreduce_norm,
            )
        elif layer_type == "*":
            self.mixer = TransformerLayer(
                model_config,
                layer_idx,
                reduce_output=has_tp_allreduce,
            )
        elif layer_type == "E":
            self.mixer = NemotronHMOE(
                model_config,
                layer_idx=layer_idx,
                aux_stream_dict=aux_stream_dict,
                reduce_output=has_tp_allreduce,
            )
        else:
            raise ValueError(f"{layer_type} is not supported")

    def post_load_weights(self):
        """Post-process after loading weights."""
        if self.norm.is_nvfp4 and not hasattr(self.norm, "nvfp4_scale"):
            self._try_attach_nvfp4_scale()

    def _try_attach_nvfp4_scale(self):
        """Attach input_scale from mixer's first linear to norm for fused RMSNorm+Quant."""
        # Normal handling for Mamba, MLP, and Attention layers.
        first_linear_attr = {
            "M": "in_proj",
            "-": "up_proj",
            "*": "qkv_proj"
        }.get(self.layer_type)
        if first_linear_attr:
            first_linear = getattr(self.mixer, first_linear_attr, None)
            if first_linear and hasattr(first_linear, "input_scale"):
                self.norm.nvfp4_scale = first_linear.input_scale
                return

        # Special handling for MoE layer: fetch shared_expert.up_proj.input_scale
        # as representation of the input scale.
        if self.is_moe_layer:
            if (hasattr(self.mixer, "shared_experts")
                    and self.mixer.shared_experts is not None
                    and hasattr(self.mixer.shared_experts, "up_proj")
                    and hasattr(self.mixer.shared_experts.up_proj,
                                "input_scale") and
                    self.mixer.shared_experts.up_proj.input_scale is not None):
                self.norm.nvfp4_scale = self.mixer.shared_experts.up_proj.input_scale
                # Enable high precision output for MoE layer.
                self.norm.return_hp_output = True
                return

        self.norm.is_nvfp4 = False
        self.norm.return_hp_output = False

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor | None = None,
        spec_metadata: SpecMetadata | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = torch.zeros_like(hidden_states)

        if hasattr(self, 'pre_allreduce'):
            norm = self.norm
            has_nvfp4_scale = hasattr(norm, 'nvfp4_scale')
            if norm.is_nvfp4 and has_nvfp4_scale and norm.return_hp_output:
                fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4
            elif norm.is_nvfp4 and has_nvfp4_scale:
                fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4
            else:
                fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
            all_reduce_params = AllReduceParams(
                fusion_op=fusion_op,
                residual=residual,
                norm_weight=norm.weight,
                eps=norm.variance_epsilon,
                trigger_completion_at_end=False,
                **(dict(scale=norm.nvfp4_scale)
                   if has_nvfp4_scale and norm.is_nvfp4 else {}),
            )
            result = self.pre_allreduce(hidden_states,
                                        all_reduce_params=all_reduce_params)
            if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
                norm_out, act_fp4, act_sf, residual = result
                hidden_states = (Fp4QuantizedTensor(act_fp4, act_sf), norm_out)
            elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
                act_fp4, act_sf, residual = result
                hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
            else:
                hidden_states, residual = result
        elif self.norm.return_hp_output:
            hidden_states, residual, high_precision_normed_output = self.norm(
                hidden_states, residual)
            hidden_states = (hidden_states, high_precision_normed_output)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        # When fuse_allreduce_norm is active, tell Transformer/MoE mixers to
        # skip their own AllReduce (it is handled by pre_allreduce /
        # final_allreduce instead).  MLP/Mamba ignore this kwarg; their
        # reduce_output was set at init time.
        mixer_kwargs = dict(spec_metadata=spec_metadata,
                            lora_params=lora_params,
                            **kwargs)
        if self.fuse_allreduce_norm:
            mixer_kwargs['all_reduce_params'] = AllReduceParams(
                enable_allreduce=False)
        hidden_states = self.mixer(hidden_states, attn_metadata, **mixer_kwargs)

        if spec_metadata is not None and spec_metadata.is_layer_capture(
                self.layer_idx):
            spec_metadata.maybe_capture_hidden_states(self.layer_idx,
                                                      hidden_states, residual)

        return hidden_states, residual


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

        self.fuse_allreduce_norm = (not model_config.mapping.enable_attention_dp
                                    and model_config.mapping.tp_size > 1)

        # create layers
        layers = []
        for layer_idx, layer_type in enumerate(config.hybrid_override_pattern):
            layers.append(
                NemotronHLayer(
                    model_config,
                    layer_idx,
                    layer_type,
                    aux_stream_dict=self.aux_stream_dict,
                    fuse_allreduce_norm=self.fuse_allreduce_norm,
                ))
        self.layers = nn.ModuleList(layers)
        self.num_hidden_layers = config.num_hidden_layers

        # final norm
        self.norm_f = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
        )

        # AllReduce for fusing with final norm (after last layer's mixer)
        if self.fuse_allreduce_norm:
            self.final_allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
            )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor | None = None,
        position_ids: torch.IntTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        spec_metadata: SpecMetadata | None = None,
        lora_params: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        mamba_metadata = attn_metadata.mamba_metadata

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = torch.zeros_like(hidden_states)
        for layer in self.layers[:self.num_hidden_layers]:
            hidden_states, residual = layer(
                position_ids,
                hidden_states,
                residual=residual,
                attn_metadata=attn_metadata,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
                mamba_metadata=mamba_metadata,
            )

        if self.fuse_allreduce_norm:
            hidden_states, _ = self.final_allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.norm_f.weight,
                    eps=self.norm_f.variance_epsilon,
                    trigger_completion_at_end=False,
                ))
        else:
            hidden_states, _ = self.norm_f(hidden_states, residual)
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

        if model_config.mapping.tp_size not in [1, 2, 4, 8]:
            raise ValueError("TP has to be either 1, 2, 4 or 8")

        if model_config.quant_config.exclude_modules is not None:
            model_config.quant_config.exclude_modules = [
                re.sub(r"(model\.layers\.)?backbone", "model", k)
                for k in model_config.quant_config.exclude_modules
            ]

        # Rename quant_config_dict keys from 'backbone.layers.' to 'model.layers.' so that
        # apply_layerwise_quant_config() can correctly match TRT-LLM module names, which use
        # 'model.layers.' as root rather than 'backbone.layers.' from the HF checkpoint.
        if model_config.quant_config_dict is not None:
            model_config._frozen = False
            model_config.quant_config_dict = {
                re.sub(r"(model\.layers\.)?backbone", "model", k): v
                for k, v in model_config.quant_config_dict.items()
            }
            model_config._frozen = True

        super().__init__(
            model=NemotronHModel(model_config),
            model_config=model_config,
        )
        self.model_nextn = 0
        if (model_config.spec_config is not None
                and model_config.spec_config.spec_dec_mode.is_mtp_one_model()):
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

    def load_weights(self, weights: dict, weight_mapper: BaseWeightMapper):
        new_weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(weights=new_weights, weight_mapper=weight_mapper)

    @classmethod
    def get_model_defaults(cls,
                           llm_args: "TorchLlmArgs",
                           pretrained_config=None) -> dict:
        """Model-specific defaults for NemotronH.

        Disables block reuse due to SSM/hybrid architecture constraints.
        """
        # TODO: Remove enable_block_reuse=False once KV cache block reuse
        # is supported for Mamba/SSM-based models
        return {"kv_cache_config": {"enable_block_reuse": False}}

    @staticmethod
    def lora_config(model_dir: str):
        """Nemotron-H-specific LoRA configuration.

        Nemotron-H is a hybrid Mamba-Attention-MoE model. The LoRA targets
        cover attention (q/k/v/o), Mamba (in/out proj), shared expert MLP.
        """
        return LoraConfig(
            lora_target_modules=[
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_dense",
                "mamba_in_proj",
                "mamba_out_proj",
                "shared_expert_h_to_4h",
                "shared_expert_4h_to_h",
                "moe_latent_fc1",
                "moe_latent_fc2",
            ],
            trtllm_modules_to_hf_modules={
                "attn_q": "q_proj",
                "attn_k": "k_proj",
                "attn_v": "v_proj",
                "attn_dense": "o_proj",
                "mamba_in_proj": "in_proj",
                "mamba_out_proj": "out_proj",
                "shared_expert_h_to_4h": "shared_experts.up_proj",
                "shared_expert_4h_to_h": "shared_experts.down_proj",
                "moe_latent_fc1": "fc1_latent_proj",
                "moe_latent_fc2": "fc2_latent_proj",
            },
        )


class NemotronHMTPDecoderLayer(NemotronHLayer):

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
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
        if (model_config.spec_config is not None
                and model_config.spec_config.spec_dec_mode.is_mtp_one_model()):
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
        attn_metadata: AttentionMetadata | None = None,
        lora_params: dict | None = None,
        **kwargs,
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
            lora_params=lora_params,
            **kwargs,
        )

        if self.has_end_norm:
            hidden_states, residual = self.final_layernorm(
                hidden_states, residual)
            # The last step, so don't forward the residual.
            residual = None

        return hidden_states, residual


class NemotronHMTP(nn.Module):
    """NemotronH MTP Layer - single MTP layer following DeepseekV3MTP pattern."""

    def __init__(
        self,
        model_config: ModelConfig[NemotronHConfig],
        layer_idx: int,
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream],
        is_separate_draft_engine: bool = False,
        prefix: str = "",
    ):
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

        for step_rel_idx in range(self.pattern_len):
            char = self.pattern_str[step_rel_idx]

            is_start_of_step = step_rel_idx == 0
            is_end_of_step = step_rel_idx == self.pattern_len - 1

            sublayer_quant_config = self._get_mtp_sublayer_quant_config(
                model_config, self.layer_idx)

            # Create a model_config copy with quant_config overridden and
            # spec_config cleared. All other fields (use_cuda_graph,
            # moe_backend, moe_max_num_tokens, etc.) must be inherited
            # so MoE layers are configured correctly for CUDA graph
            # capture and communication (e.g., DeepEP).
            sublayer_model_config = replace(model_config,
                                            quant_config=sublayer_quant_config,
                                            spec_config=None)

            self.layers[str(step_rel_idx)] = NemotronHMTPDecoderLayer(
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
        all_rank_num_tokens: list[int] | None = None,
        spec_metadata: SpecMetadata | None = None,
        lora_params: dict | None = None,
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
                all_rank_num_tokens=all_rank_num_tokens,
                lora_params=lora_params,
            )
        return hidden_states


AutoConfig.register(NemotronHConfig.model_type, NemotronHConfig)
