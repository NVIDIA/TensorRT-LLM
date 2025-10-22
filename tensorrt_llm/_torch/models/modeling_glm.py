import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import PretrainedConfig

from tensorrt_llm._ipc_utils import can_access_peer
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.quantization.utils.fp8_utils import (
    resmooth_to_fp8_e8m0,
    transform_sf_into_required_layout,
)

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    MoEAllReduce,
    MoEAllReduceParams,
)
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import MoEWeightLoadingMode, create_moe
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.qk_norm_attention import QKNormRoPEAttention
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType, Fp4QuantizedTensor
from .modeling_deepseekv3 import DeepseekV3Gate, DeepseekV3MTPHead, moe_reduce_add_shared_output
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, EagerFusionConfig, _load_weights_impl, register_auto_model


class Glm4Attention(QKNormRoPEAttention):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
    ):
        config = model_config.pretrained_config
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.yarn,
            rope=RopeParams.from_config(config),
        )

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=config.attention_bias,
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            dense_bias=False,
            config=model_config,
        )


class Glm4MoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        shared_expert_intermediate_size: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        dtype: Optional[torch.dtype] = None,
        model_config: ModelConfig = ModelConfig(),
        override_quant_config: Optional[QuantConfig] = None,
        layer_idx: Optional[int] = None,
    ):
        from ..distributed import AllReduce

        super().__init__()
        config = model_config.pretrained_config
        self.top_k = top_k
        self.use_dp = model_config.mapping.enable_attention_dp
        self.gate = DeepseekV3Gate(
            hidden_size,
            num_experts,
            top_k=top_k,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            dtype=dtype,
            fuse_routing_kernel=False,
            apply_routing=False,
            moe_backend=model_config.moe_backend,
        )
        self.experts = create_moe(
            num_experts=num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=False,  # In both low‑latency and attention‑DP modes, FusedMoE skips the in‑op all‑reduce.
            model_config=model_config,
            override_quant_config=override_quant_config,
            aux_stream_dict=aux_stream_dict,
            layer_idx=layer_idx,
            weight_loading_mode=MoEWeightLoadingMode.VANILLA,
        )

        self.mapping = model_config.mapping

        # FIXME: incompatible with mixed quantization mode (including excluding modules from quantization)
        block_size = 1
        if (
            model_config.quant_config
            and model_config.quant_config.quant_algo
            and model_config.quant_config.group_size is not None
        ):
            block_size = model_config.quant_config.group_size

        shared_tp_size, self.shared_output_scale = self._compute_shared_expert_tp_size(
            shared_expert_intermediate_size, block_size
        )

        self.shared_experts = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=model_config,
            overridden_tp_size=shared_tp_size,
            reduce_output=False,
        )

        self.allreduce = AllReduce(
            mapping=model_config.mapping, strategy=model_config.allreduce_strategy
        )
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

    def _compute_shared_expert_tp_size(
        self, intermediate_size: int, block_size: int
    ) -> tuple[int, float | None]:
        """
        In the case of GLM4, the TP size of MLP is capped by intermediate_size // block_size.
        For example, when the intermediate_size is 2048 and block scaling size is 128,
        TP sizes are limited to {1, 2, 4, 8, 16} because of 2048/128 = 16.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. For NVFP4, it's 16.

        Returns:
            tuple[int, float | None]: A tuple containing (shared_tp_size, shared_output_scale).
                - shared_tp_size: The computed TP size.
                - shared_output_scale: The output scale factor, or None if not needed.
        """

        assert intermediate_size % block_size == 0, (
            "intermediate_size must be divisible by block_size."
        )

        shared_output_scale = None
        # The block scale size is 128, which requires shared_expert_intermediate_size to be divisible by 128.
        if self.use_dp:
            # If using attention DP, the shared experts also use DP instead of TP.
            shared_tp_size = 1
        else:
            # Due to the restriction of block scale size (i.e., 128),
            # the supported TP sizes only include 1, 2, 4, 8, and 16.
            # The math.gcd operation ensures that shared_tp_size falls in the supported TP sizes.
            shared_tp_size = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )
            # If shared_tp_size has been overridden, the output of shared experts needs to be
            # scaled down accordingly before all-reduce.
            if shared_tp_size != self.mapping.tp_size:
                shared_output_scale = shared_tp_size / self.mapping.tp_size

        return shared_tp_size, shared_output_scale

    @staticmethod
    def _get_experts_quant_config(model_config, layer_idx: int) -> QuantConfig:
        if getattr(model_config, "quant_config_dict", None) is None:
            return model_config.quant_config
        return model_config.quant_config_dict.get(
            f"model.layers.{layer_idx}.mlp.experts", model_config.quant_config
        )

    def compute_routed_output(
        self, hidden_states, hidden_states_fp4, all_rank_num_tokens, do_finalize
    ):
        # max-throughput
        use_dp_padding = False
        # Add DP padding on SM120 for context comm performance
        # TODO: Move this model-agonostic part to MoE
        if self.use_dp and self.mapping.tp_size > 1 and get_sm_version() == 120:
            use_dp_padding = True
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, max(all_rank_num_tokens) - hidden_states.shape[0])
            )

        router_logits = self.gate(hidden_states)

        routed_output = self.experts(
            hidden_states_fp4 if hidden_states_fp4 is not None else hidden_states,
            router_logits,
            do_finalize=do_finalize,
            output_dtype=hidden_states.dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        return routed_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Optional[Fp4QuantizedTensor] = None,
        all_rank_num_tokens: Optional[list[int]] = None,
        final_all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
    ) -> torch.Tensor:
        if not do_finalize:
            assert not self.use_dp

        def _compute_shared_output():
            shared_output = self.shared_experts(
                hidden_states_fp4 if hidden_states_fp4 is not None else hidden_states
            )
            if self.shared_output_scale is not None:
                shared_output *= self.shared_output_scale
            return shared_output

        def _compute_routed_output():
            routed_output = self.compute_routed_output(
                hidden_states, hidden_states_fp4, all_rank_num_tokens, do_finalize
            )
            return routed_output

        # NOTE: define compiled helpers at module scope to avoid defining decorators inside compiled frames

        routed_output, shared_output = maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
        )

        if not do_finalize:
            return [shared_output, *routed_output]
        else:
            if routed_output.dim() == 3:
                assert shared_output.numel() * self.top_k == routed_output.numel(), (
                    "unmatched tensor shape"
                )
                final_hidden_states = moe_reduce_add_shared_output(routed_output, shared_output)
            else:
                assert shared_output.size() == routed_output.size(), "unmatched tensor shape"
                final_hidden_states = shared_output + routed_output

            if not self.use_dp and self.mapping.tp_size > 1:
                final_hidden_states = self.allreduce(
                    final_hidden_states, all_reduce_params=final_all_reduce_params
                )

            return final_hidden_states


class Glm4DecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        is_separate_draft_engine: bool = False,
    ):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config
        config = self.config

        self.hidden_size = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.mapping = model_config.mapping
        mapping = self.mapping
        layer_idx_for_attention = layer_idx
        if is_separate_draft_engine:
            # KVCacheManager only support 1 layer for separate draft engine
            layer_idx_for_attention = layer_idx - model_config.pretrained_config.num_hidden_layers

        self.self_attn = Glm4Attention(
            model_config,
            layer_idx=layer_idx_for_attention,
        )
        self.enable_attention_dp = mapping.enable_attention_dp

        self.mlp_tp_size = mapping.tp_size
        self.is_p2p_supported = can_access_peer(mapping)

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get("TRTLLM_GLM_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        # FIXME: incompatible with mixed quantization mode
        quant_config = self._get_decoder_layer_quant_config(model_config, layer_idx)
        self.is_nvfp4 = quant_config.layer_quant_mode.has_nvfp4()
        assert quant_config.quant_algo is not QuantAlgo.MIXED_PRECISION, (
            "MIXED_PRECISION is ambiguous"
        )

        has_tp = mapping.has_tp()
        self.allreduce = AllReduce(
            mapping=model_config.mapping,
            strategy=model_config.allreduce_strategy,
            dtype=config.torch_dtype,
        )
        self.moe_allreduce = MoEAllReduce(self.mapping)

        if config.n_routed_experts is not None and layer_idx >= config.first_k_dense_replace:
            self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
            self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION

            self.mlp = Glm4MoE(
                num_experts=self.num_experts,
                top_k=self.top_k,
                hidden_size=self.hidden_size,
                intermediate_size=self.moe_intermediate_size,
                shared_expert_intermediate_size=self.moe_intermediate_size
                * self.num_shared_experts,
                dtype=config.torch_dtype,
                model_config=model_config,
                override_quant_config=quant_config,
                aux_stream_dict=aux_stream_dict,
                layer_idx=layer_idx,
            )
        else:
            block_size = 1
            if quant_config and quant_config.quant_algo and quant_config.group_size is not None:
                block_size = quant_config.group_size
            self.mlp_tp_size = self._compute_mlp_tp_size(config.intermediate_size, block_size)

            has_mlp_tp = self.mlp_tp_size > 1
            self.fusion_config.PRE_MLP_FUSION = self.enable_fusion and has_mlp_tp and self.is_nvfp4
            self.fusion_config.POST_MLP_FUSION = self.enable_fusion and has_mlp_tp

            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                overridden_tp_size=self.mlp_tp_size,
                reduce_output=True,
            )

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        self.disable_attn_allreduce = (
            self.fusion_config.PRE_MOE_FUSION
            or self.fusion_config.PRE_MLP_FUSION
            or self.mapping.tp_size == 1
            or self.enable_attention_dp
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.layer_idx = layer_idx
        self.next_layer_layernorm: RMSNorm = None

    def _get_decoder_layer_quant_config(
        self, model_config: ModelConfig[PretrainedConfig], layer_idx: int
    ):
        """
        The MTP layer in the nvfp4 checkpoint is unquantized. Because the TRTLLM
        moe_backend only supports fp8/fp4 quantization, we need to override
        the quant_config for the MTP layer.
        """
        quant_config = model_config.quant_config

        layer_name = f"model.layers.{layer_idx}"
        if quant_config.is_module_excluded_from_quantization(layer_name):
            return QuantConfig(
                quant_algo=None,
                kv_cache_quant_algo=quant_config.kv_cache_quant_algo,
            )
        else:
            return model_config.quant_config

    def _compute_mlp_tp_size(self, intermediate_size: int, block_size: int) -> int:
        """
        For GLM4, MLP TP size is limited by intermediate_size // block_size
        and must also be multiples of gpus_per_node to avoid expensive inter‑node allreduce.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. For NVFP4, it's 16.

        Returns:
            int: The computed tp_size.
        """

        assert intermediate_size % block_size == 0, (
            "intermediate_size must be divisible by block_size."
        )
        if self.enable_attention_dp:
            # If using attention DP, the MLP also uses DP instead of TP.
            mlp_tp_size = 1
        else:
            # The two math.gcd operations ensure that mlp_tp_size falls in the candidate TP sizes.
            tp = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )

            if tp > self.mapping.gpus_per_node:
                mlp_tp_size = math.gcd(
                    tp,
                    self.mapping.gpus_per_node,
                )  # Avoid costly inter-node TP
            else:
                mlp_tp_size = tp
        return mlp_tp_size

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )
        if isinstance(self.mlp, Glm4MoE):
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                self.fusion_config.POST_MOE_FUSION = False
            return self.forward_MoE(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                self.fusion_config.POST_MLP_FUSION = False
            assert isinstance(self.mlp, GatedMLP)
            return self.forward_mlp(
                hidden_states=hidden_states,
                residual=residual,
                spec_metadata=spec_metadata,
            )

    def forward_MoE(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _run_MoE(hidden_states, hidden_states_fp4, do_finalize):
            return self.mlp(
                hidden_states,
                hidden_states_fp4,
                all_rank_num_tokens=attn_metadata.all_rank_num_tokens,
                final_all_reduce_params=AllReduceParams(
                    enable_allreduce=not (
                        self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                    )
                ),
                do_finalize=do_finalize,
            )

        if self.fusion_config.PRE_MOE_FUSION:
            # moe_backend can be either CUTLASS or TRTLLM here
            # TODO: unify the two min-latency MoE backends by enabling quant fusion
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    trigger_completion_at_end=False,
                ),
            )
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Note: this fusion pattern is only supported for single-node TRTLLM-nvfp4 backend now
        do_finalize = self.mapping.is_multi_node() or (
            not (
                hidden_states.shape[0] <= self.moe_allreduce.max_token
                and self.fusion_config.POST_MOE_FUSION
                and self.model_config.moe_backend == "TRTLLM"
                and self.mlp.experts.has_nvfp4
                and self.is_p2p_supported
            )
        )

        hidden_states = _run_MoE(hidden_states, hidden_states_fp4=None, do_finalize=do_finalize)

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
                hidden_states, residual = self.allreduce(
                    hidden_states,
                    all_reduce_params=AllReduceParams(
                        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                        residual=residual,
                        norm_weight=self.next_layer_layernorm.weight,
                        eps=self.next_layer_layernorm.variance_epsilon,
                        trigger_completion_at_end=False,
                    ),
                )
            else:
                assert len(hidden_states) == 4, "hidden_states must have 4 elements"

                shared_output = hidden_states[0]
                fc2_output = hidden_states[1]
                expert_scale_factor = hidden_states[2]
                expanded_idx_to_permuted_idx = hidden_states[3]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=shared_output,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params
                )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(hidden_states, residual)

        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.fusion_config.PRE_MLP_FUSION:
            act_fp4, act_sf, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    scale=self.mlp.gate_up_proj.input_scale,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ),
            )
            hidden_states = Fp4QuantizedTensor(act_fp4, act_sf)
        else:
            # No fusion
            # We need to add twoshot allreduce here to avoid modifying MLA logic
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(
            hidden_states,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not (self.fusion_config.POST_MLP_FUSION or self.mlp_tp_size == 1)
            ),
        )

        if self.fusion_config.POST_MLP_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                ),
            )
        else:
            if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(hidden_states, residual)

        return hidden_states, residual


class Glm4MTP(Glm4DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        is_separate_draft_engine: bool = False,
    ):
        super().__init__(model_config, layer_idx, aux_stream_dict, is_separate_draft_engine)
        config = model_config.pretrained_config
        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok

        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

        self.enorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        self.hnorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        if model_config.mapping.enable_attention_dp:
            self.eh_proj = Linear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )
        else:
            self.eh_proj = Linear(
                config.hidden_size * 2,
                config.hidden_size,
                bias=False,
                dtype=config.torch_dtype,
                tensor_parallel_mode=TensorParallelMode.ROW,
                mapping=model_config.mapping,
                reduce_output=True,
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            )

        self.shared_head = DeepseekV3MTPHead(model_config)

    def forward(
        self,
        input_ids: torch.IntTensor,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        embed_tokens: Embedding,
        attn_metadata: AttentionMetadata,
        all_rank_num_tokens: Optional[List[int]] = None,
        **kwargs,
    ) -> torch.Tensor:
        def norm_embeds():
            return self.enorm(embed_tokens(input_ids))  # emdedding

        def norm_hidden():
            return self.hnorm(hidden_states)

        inputs_embeds, hidden_states = maybe_execute_in_parallel(
            norm_embeds,
            norm_hidden,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
        )
        hidden_states = torch.concat([inputs_embeds, hidden_states], dim=-1)
        # Split hidden_states columnwise based on TP
        tp_size = self.model_config.mapping.tp_size
        tp_rank = self.model_config.mapping.tp_rank

        if tp_size > 1 and not (self.model_config.mapping.enable_attention_dp):
            hidden_states = torch.chunk(hidden_states, tp_size, dim=-1)[tp_rank]
        hidden_states = self.eh_proj(hidden_states)

        # Input layer norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )

        # MTP Layer Must have sparse MOE
        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                ),
            )
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MoE
        hidden_states = self.mlp(
            hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=not (
                    self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                )
            ),
        )

        if self.fusion_config.POST_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.shared_head.norm.weight,
                    eps=self.shared_head.norm.variance_epsilon,
                ),
            )
        else:
            hidden_states, _ = self.shared_head.norm(hidden_states, residual)

        return hidden_states


class Glm4Model(DecoderModel):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model_config)
        config = model_config.pretrained_config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
        }

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
        )

        self.layers = nn.ModuleList(
            [
                Glm4DecoderLayer(model_config, layer_idx, self.aux_stream_dict)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None

        for decoder_layer in self.layers[: self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
            )

        return hidden_states


@register_auto_model("Glm4MoeForCausalLM")
class Glm4MoeForCausalLM(SpecDecOneEngineForCausalLM[Glm4Model, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__(model=Glm4Model(model_config), model_config=model_config)

        self.model_nextn = 0
        if (
            model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
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
                        self.num_hidden_layers, self.num_hidden_layers + model_nextn
                    ):
                        ckpt_mtp_idx = (
                            model_mtp_idx - self.num_hidden_layers
                        ) % ckpt_nextn + self.num_hidden_layers
                        model_prefix = f"model.layers.{model_mtp_idx}"
                        ckpt_prefix = f"model.layers.{ckpt_mtp_idx}"
                        for exclude_module in model_config.quant_config.exclude_modules:
                            if ckpt_prefix in exclude_module and model_prefix not in exclude_module:
                                extend_exclude_modules.append(
                                    exclude_module.replace(ckpt_prefix, model_prefix)
                                )
                    self.model_config.quant_config.exclude_modules.extend(extend_exclude_modules)
            self.model.layers.extend(self.draft_model.mtp_layers)
            self.epilogue.extend(self.draft_model.mtp_layers)
            self.epilogue.append(self.spec_worker)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: torch.IntTensor = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            spec_metadata=spec_metadata,
            return_context_logits=return_context_logits,
            **kwargs,
        )

    def load_weights(self, weights: Dict):
        # model.layers.91.mlp.experts.75.up_proj.weight_scale_2
        _load_weights_impl(
            self,
            weights,
            params_map={
                r"(?!.*shared_experts)(?=.*experts?)(.*?)up_proj(.*)": r"\1w3\2",
                r"(?!.*shared_experts)(?=.*experts?)(.*?)down_proj(.*)": r"\1w2\2",
                r"(?!.*shared_experts)(?=.*experts?)(.*?)gate_proj(.*)": r"\1w1\2",
            },
        )

    def post_load_weights(self):
        all_named_modules = dict(self.model.named_modules())
        for name, module in tqdm(all_named_modules.items(), desc="Post loading weights"):
            if len(module._parameters) <= 0 or name.startswith("draft_model"):
                continue
            else:
                if (
                    self.model_config.quant_config.layer_quant_mode.has_fp8_block_scales()
                    and is_sm_100f()
                    and hasattr(module, "weight_scale")
                ):
                    weight, weight_scale = resmooth_to_fp8_e8m0(module.weight, module.weight_scale)
                    transfromed_scale = transform_sf_into_required_layout(
                        weight_scale,
                        mn=weight.shape[0],
                        k=weight.shape[1],
                        recipe=(1, 128, 128),
                        is_sfa=False,
                    )
                    module.weight = nn.Parameter(weight, requires_grad=False)
                    module.weight_scale = nn.Parameter(transfromed_scale, requires_grad=False)

        for idx, layer in enumerate(self.model.layers[: self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[idx + 1].input_layernorm
