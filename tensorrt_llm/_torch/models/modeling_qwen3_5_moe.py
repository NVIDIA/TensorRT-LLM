import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_disagg
from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata

from ...inputs import (
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
    support_multimodal_disaggregated,
)
from ..attention_backend import AttentionMetadata
from ..distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    MoEAllReduce,
    MoEAllReduceParams,
    allgather,
)
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.fused_moe import RoutingMethodType, TRTLLMGenFusedMoE, create_moe
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..utils import AuxStreamType, EventType
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3_5_moe_weight_mapper import Qwen3_5MoeHfWeightMapper, Qwen3_5MoeTextConfig
from .modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5InputProcessorBase,
    Qwen3_5ModelBase,
    Qwen3_5VisionModel,
    Qwen3_5VisionModelBase,
)
from .modeling_qwen3_next import Qwen3NextGate
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import (
    DecoderModel,
    EagerFusionConfig,
    ModelConfig,
    SpecMetadata,
    register_auto_model,
    register_vision_encoder,
)


class Qwen3_5MoeGate(Qwen3NextGate):
    pass


class Qwen3_5SparseMoeBlock(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[Qwen3_5MoeTextConfig],
        aux_stream: torch.cuda.Stream,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        config = model_config.pretrained_config
        self.model_config = model_config
        self.hidden_dim = config.hidden_size
        self.moe_intermediate_size = config.moe_intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.enable_attention_dp = model_config.mapping.enable_attention_dp
        self.mapping = model_config.mapping
        self.allreduce = AllReduce(
            mapping=model_config.mapping, strategy=model_config.allreduce_strategy
        )
        self.aux_stream = aux_stream

        self.gate = Qwen3_5MoeGate(
            hidden_size=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=self.top_k,
            dtype=config.torch_dtype,
            apply_routing=False,
            routing_method_type=RoutingMethodType.Renormalize,
            moe_backend=model_config.moe_backend,
        )

        self.experts = create_moe(
            num_experts=self.num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=self.hidden_dim,
            intermediate_size=self.moe_intermediate_size,
            aux_stream_dict={AuxStreamType.MoeChunkingOverlap: aux_stream},
            dtype=config.torch_dtype,
            reduce_results=False,
            model_config=model_config,
            layer_idx=layer_idx,
        )

        self.shared_expert = GatedMLP(
            hidden_size=self.hidden_dim,
            intermediate_size=config.shared_expert_intermediate_size,
            bias=config.mlp_bias if hasattr(config, "mlp_bias") else False,
            dtype=config.torch_dtype,
            config=model_config,
            reduce_output=False,
        )

        self.shared_expert_gate = Linear(
            self.hidden_dim, 1, bias=False, dtype=config.torch_dtype, quant_config=None
        )

        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        do_finalize: Optional[bool] = True,
    ) -> torch.Tensor:
        assert hidden_states.shape[-1] == self.hidden_dim
        orig_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_dim)
        use_dp_padding = False
        all_rank_num_tokens = attn_metadata.all_rank_num_tokens

        if not do_finalize:
            # TODO: support do_finalize == False
            raise NotImplementedError("do_finalize == False is not supported yet")

        if self.enable_attention_dp and self.mapping.tp_size > 1:
            if isinstance(self.experts, TRTLLMGenFusedMoE):
                hidden_states = allgather(
                    hidden_states, self.mapping, dim=0, sizes=all_rank_num_tokens
                )

        def _compute_routed_output():
            router_logits = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states,
                router_logits,
                all_rank_num_tokens=all_rank_num_tokens,
                use_dp_padding=use_dp_padding,
                do_finalize=do_finalize,
            )
            return final_hidden_states

        def _compute_shared_output():
            shared_expert_output = self.shared_expert(hidden_states)
            shared_expert_output = (
                F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
            )
            return shared_expert_output

        final_hidden_states, shared_expert_output = maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
        )
        if not do_finalize:
            return final_hidden_states

        final_hidden_states = final_hidden_states + shared_expert_output

        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            final_hidden_states = self.allreduce(
                final_hidden_states, all_reduce_params=all_reduce_params
            )

        return final_hidden_states.view(orig_shape)


class Qwen3_5MoeLinearDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[Qwen3_5MoeTextConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config
        self.linear_attn = Qwen3_5GatedDeltaNet(model_config, aux_stream, layer_idx)

        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = Qwen3_5SparseMoeBlock(model_config, aux_stream, layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(
            mapping=model_config.mapping, strategy=model_config.allreduce_strategy
        )
        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        ### TODO: enable eager_fusion by default
        self.enable_fusion = os.environ.get("TRTLLM_QWEN3_5_EAGER_FUSION_DISABLED", "1") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        # has_tp = self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        # self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        self.fusion_config.PRE_MOE_FUSION = (
            False  # the fusion kernel does not support gemmaNorm yet
        )
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        self.disable_attn_allreduce = (
            self.fusion_config.PRE_MOE_FUSION
            or self.mapping.tp_size == 1
            or self.enable_attention_dp
        )
        self.moe_allreduce = MoEAllReduce(mapping=model_config.mapping)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False
        # Linear Attention
        ### FIXME: 1. forward_batch; 2. allreduce
        if hidden_states.shape[0] != 0:
            hidden_states = self.linear_attn(
                hidden_states,
                attn_metadata,
                all_reduce_params=AllReduceParams(enable_allreduce=not self.disable_attn_allreduce),
                **kwargs,
            )
        if self.fusion_config.PRE_MOE_FUSION:
            hidden_states, residual = self.allreduce(
                hidden_states,
                all_reduce_params=AllReduceParams(
                    fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                    residual=residual,
                    norm_weight=self.post_attention_layernorm.weight,
                    eps=self.post_attention_layernorm.variance_epsilon,
                    enable_allreduce=not (
                        self.fusion_config.PRE_MOE_FUSION or self.mapping.tp_size == 1
                    ),
                ),
            )
        else:
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Note: this fusion pattern is only supported for TRTLLM-nvfp4 backend now
        do_finalize = not (
            hidden_states.shape[0] <= self.moe_allreduce.max_token
            and self.fusion_config.POST_MOE_FUSION
            and self.model_config.moe_backend == "TRTLLM"
            and self.mlp.experts.has_nvfp4
        )

        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (
                    self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                )
            ),
            do_finalize=do_finalize,
        )
        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
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
                assert len(hidden_states) == 3, (
                    f"hidden_states must have 3 elements, but got {len(hidden_states)}"
                )

                fc2_output = hidden_states[0]
                expert_scale_factor = hidden_states[1]
                expanded_idx_to_permuted_idx = hidden_states[2]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=None,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params
                )

        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(hidden_states, residual)
        return hidden_states, residual


class Qwen3_5MoeFullAttentionDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[Qwen3_5MoeTextConfig],
        layer_idx: int,
        aux_stream: torch.cuda.Stream,
    ):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config

        self.self_attn = Qwen3_5Attention(
            model_config,
            layer_idx=layer_idx,
            fuse_qk_norm_rope=False,
        )
        self.mapping = model_config.mapping
        self.enable_attention_dp = self.mapping.enable_attention_dp

        self.mlp = Qwen3_5SparseMoeBlock(model_config, aux_stream, layer_idx=layer_idx)

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=config.torch_dtype,
            use_gemma=True,
        )
        self.layer_idx = layer_idx

        self.allreduce = AllReduce(
            mapping=model_config.mapping, strategy=model_config.allreduce_strategy
        )
        self.next_layer_layernorm: RMSNorm = None

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get("TRTLLM_QWEN3_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        # has_tp = self.mapping.has_tp()
        has_pp = self.mapping.has_pp()

        # self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
        self.fusion_config.PRE_MOE_FUSION = False
        self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
        self.disable_attn_allreduce = (
            self.fusion_config.PRE_MOE_FUSION
            or self.mapping.tp_size == 1
            or self.enable_attention_dp
        )
        self.moe_allreduce = MoEAllReduce(mapping=model_config.mapping)

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        if spec_metadata is not None and spec_metadata.is_layer_capture(self.layer_idx):
            self.fusion_config.POST_MOE_FUSION = False

        # Self Attention
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not self.disable_attn_allreduce),
            **kwargs,
        )

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
            # No fusion
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # Note: this fusion pattern is only supported for TRTLLM-nvfp4 backend now
        do_finalize = not (
            hidden_states.shape[0] <= self.moe_allreduce.max_token
            and self.fusion_config.POST_MOE_FUSION
            and self.model_config.moe_backend == "TRTLLM"
            and self.mlp.experts.has_nvfp4
        )

        hidden_states = self.mlp(
            hidden_states,
            attn_metadata,
            all_reduce_params=AllReduceParams(
                enable_allreduce=not (
                    self.fusion_config.POST_MOE_FUSION or self.mapping.tp_size == 1
                )
            ),
            do_finalize=do_finalize,
        )

        if self.fusion_config.POST_MOE_FUSION:
            if do_finalize:
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
                assert len(hidden_states) == 3, (
                    f"hidden_states must have 3 elements, but got {len(hidden_states)}"
                )

                fc2_output = hidden_states[0]
                expert_scale_factor = hidden_states[1]
                expanded_idx_to_permuted_idx = hidden_states[2]

                moe_all_reduce_params = MoEAllReduceParams(
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    expert_scale_factor=expert_scale_factor,
                    shared_expert_output=None,
                    residual=residual,
                    norm_weight=self.next_layer_layernorm.weight,
                    eps=self.next_layer_layernorm.variance_epsilon,
                    is_cutlass_min_latency=False,
                )
                hidden_states, residual = self.moe_allreduce(
                    fc2_output, all_reduce_params=moe_all_reduce_params
                )

        else:
            if spec_metadata and spec_metadata.is_layer_capture(self.layer_idx):
                spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
            if self.next_layer_layernorm is not None:
                hidden_states, residual = self.next_layer_layernorm(hidden_states, residual)

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "full_attention": Qwen3_5MoeFullAttentionDecoderLayer,
    "linear_attention": Qwen3_5MoeLinearDecoderLayer,
}


class Qwen3_5MoeTextModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[Qwen3_5MoeTextConfig]):
        super().__init__(model_config)
        config = self.model_config
        pretrained_config = self.model_config.pretrained_config
        self.aux_stream = torch.cuda.Stream()
        self.preload_weight_modules = []
        if config.moe_backend == "TRTLLM":
            self.preload_weight_modules = [
                "experts",
                "routing_method",
                "all_reduce",
            ]

        if model_config.mapping.enable_attention_dp:
            # When attention_dp is enabled, we cannot do all_reduce since
            # the problem size of different ranks are different.
            # So, we don't do parallelism here.
            self.embed_tokens = Embedding(
                pretrained_config.vocab_size,
                pretrained_config.hidden_size,
                dtype=pretrained_config.torch_dtype,
            )
        else:
            self.embed_tokens = Embedding(
                pretrained_config.vocab_size,
                pretrained_config.hidden_size,
                dtype=pretrained_config.torch_dtype,
                mapping=config.mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )

        self.layers = nn.ModuleList(
            [
                ALL_DECODER_LAYER_TYPES[pretrained_config.layer_types[layer_idx]](
                    model_config,
                    layer_idx,
                    self.aux_stream,
                )
                for layer_idx in range(pretrained_config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=pretrained_config.hidden_size,
            eps=pretrained_config.rms_norm_eps,
            dtype=pretrained_config.torch_dtype,
            use_gemma=True,
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

        mamba_metadata = attn_metadata.mamba_metadata
        if mamba_metadata.max_batch_size != attn_metadata.max_num_requests:
            attn_metadata.mamba_metadata = Mamba2Metadata(
                attn_metadata.max_num_requests, chunk_size=128
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                mamba_metadata=mamba_metadata,
            )
        return hidden_states


@register_auto_model("Qwen3_5MoeForCausalLM")
class Qwen3_5MoeForCausalLM(SpecDecOneEngineForCausalLM[Qwen3_5MoeTextModel, Qwen3_5MoeTextConfig]):
    def __init__(
        self,
        model_config: ModelConfig[Qwen3_5MoeTextConfig],
    ):
        super().__init__(
            Qwen3_5MoeTextModel(model_config),
            model_config,
        )
        self.preload_weight_modules = self.model.preload_weight_modules

    def load_weights(
        self,
        weights: dict,
        weight_mapper: BaseWeightMapper,
        params_map: Optional[Dict[str, str]] = None,
    ):
        new_weights = weight_mapper.preprocess_weights(weights)
        super().load_weights(new_weights, weight_mapper, params_map=params_map)

    def post_load_weights(self):
        for idx, layer in enumerate(self.model.layers[: self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[idx + 1].input_layernorm


# NOTE: this is technically not strictly necessary, since the underlying mechanism for registering
# support is tacked onto the input processor class (`Qwen3_5InputProcessorBase`). Given that
# the `Qwen_5Model` (defined via the import of `modeling_qwen3_5.py` in this file) has that
# decorator applied to it, and uses the same input processor class, we get it "for free" here.
# However, we keep it here to explicitly signify intent that this is supported. This also shields
# it from e.g. the input processor classes becoming specialized between `Qwen3_5Model` and the
# below MoE class.
@support_multimodal_disaggregated
@register_vision_encoder(Qwen3_5VisionModelBase, vlm_base_model=Qwen3_5VisionModel)
@register_auto_model("Qwen3_5MoeForConditionalGeneration")
@register_input_processor(
    Qwen3_5InputProcessorBase,
    model_type="qwen3_5_moe",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    ),
)
class Qwen3_5MoeModel(Qwen3_5ModelBase):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        # NOTE: HF implementation.
        kwargs["vision_model_class"] = Qwen3_5VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get(
            "disable_fuse_rope", False
        )  # TODO: Make this ModelConfig's argument
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "multimodal_embedding",
        ]

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        if not _is_disagg():
            self.mm_encoder.load_weights(weights)

        weight_mapper = Qwen3_5MoeHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        filtered_weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
        params_map = {
            r"^model\.language_model\.(.*)$": r"model.\1",
        }
        self.llm.load_weights(filtered_weights, weight_mapper, params_map=params_map)
