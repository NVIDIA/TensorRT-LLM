import math
import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._ipc_utils import can_access_peer
from tensorrt_llm._torch.modules.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization import QuantAlgo

from ...logger import logger
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    MoEAllReduce,
    MoEAllReduceParams,
)
from ..model_config import ModelConfig
from ..models.modeling_deepseekv3 import Deepseekv3MoE
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType, EventType, Fp4QuantizedTensor
from .checkpoints.hf.exaone_moe_weight_mapper import ExaoneMoeWeightMapper
from .modeling_deepseekv3 import DeepseekV3MTPHead
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import DecoderModel, EagerFusionConfig, register_auto_model

# fmt: off
# TODO: Remove this once we have a proper transformers package
from transformers import AutoConfig, PretrainedConfig  # isort: skip

class ExaoneMoEConfig(PretrainedConfig):
    model_type = "exaone_moe"

logger.warning_once(
    "transformers does not support 'ExaoneMoEConfig'. "
    "Register ExaoneMoEConfig to mimic the ExaoneMoE model.",
    key="EXAONE_MOE_REGISTER_WARNING"
)
AutoConfig.register(ExaoneMoEConfig.model_type, ExaoneMoEConfig)
# End of the config register.
# fmt: on


def check_is_moe(config: ExaoneMoEConfig, layer_idx: int, is_mtp_layer: bool = False) -> bool:
    """
    Check if the current layer is a MoE layer.
    """
    return not is_mtp_layer and hasattr(config, "is_moe_layer") and config.is_moe_layer[layer_idx]


def enable_attn_allreduce(mapping: Mapping):
    return not mapping.enable_attention_dp or mapping.has_tp()


class ExaoneMoeAttention(QKNormRoPEAttention):
    def __init__(
        self,
        model_config: ModelConfig[ExaoneMoEConfig],
        layer_idx: Optional[int] = None,
        is_mtp_layer: bool = False,
        fuse_qk_norm_rope: bool = False,
        disable_deep_gemm: bool = False,
    ):
        config = model_config.pretrained_config

        self.attention_window_size = None
        # A MTP layer uses the global attention.
        self.is_sliding = not is_mtp_layer and config.layer_types[layer_idx] == "sliding_attention"

        # NOTE: In ExaoneMoe, only sliding layers apply rope.
        pos_embd_params = None
        if self.is_sliding:
            self.attention_window_size = config.sliding_window
            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )

        fuse_qk_norm_rope = self.is_sliding and fuse_qk_norm_rope

        # NOTE: Fusing qk norm with rope has an issue that slightly hurts accuracy.
        assert not fuse_qk_norm_rope, "Fusing qk norm and rope is having issue now"

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            skip_rope=not self.is_sliding,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            disable_deep_gemm=disable_deep_gemm,
            reduce_output=enable_attn_allreduce(model_config.mapping),
        )

    def forward(
        self,
        position_ids: Optional[torch.LongTensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        attention_mask: PredefinedAttentionMask = PredefinedAttentionMask.CAUSAL,
        lora_params: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            attention_mask=attention_mask,
            lora_params=lora_params,
            attention_window_size=self.attention_window_size,
            **kwargs,
        )


class ExaoneMoeSparseMoEBlock(Deepseekv3MoE):
    """
    ExaoneMoe Sparse MoE Block Layer.

    It follows DeepSeek-V3 implementation.
    """


class ExaoneMoeDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[ExaoneMoEConfig],
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        layer_idx: int,
    ):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config
        self.layer_idx = layer_idx

        self.mapping = model_config.mapping
        mapping = self.mapping
        self.enable_attention_dp = mapping.enable_attention_dp
        self.mlp_tp_size = mapping.tp_size
        self.is_p2p_supported = can_access_peer(mapping)

        self.fusion_config = EagerFusionConfig()
        # MoE fusions are disabled by default in K-EXAONE since
        # it may cause a slight accuracy drop due to numerical gap.
        self.enable_fusion = os.environ.get("TRTLLM_EXAONE_EAGER_FUSION_ENABLED", "0") == "1"
        self.enable_fusion &= not self.enable_attention_dp

        # FIXME: incompatible with mixed quantization mode
        quant_config = self._get_decoder_layer_quant_config(model_config, layer_idx)
        self.is_nvfp4 = quant_config.layer_quant_mode.has_nvfp4()
        assert quant_config.quant_algo is not QuantAlgo.MIXED_PRECISION, (
            "MIXED_PRECISION is ambiguous"
        )

        self.allreduce = None
        self.moe_allreduce = None
        if not self.enable_attention_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping,
                strategy=model_config.allreduce_strategy,
                dtype=config.torch_dtype,
            )
            self.moe_allreduce = MoEAllReduce(self.mapping)

        has_tp = mapping.has_tp()
        has_pp = mapping.has_pp()

        # Submodule definitions
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

        is_mtp_layer = False
        if layer_idx >= config.num_hidden_layers:
            is_mtp_layer = True
        self.self_attn = ExaoneMoeAttention(
            model_config, layer_idx=layer_idx, is_mtp_layer=is_mtp_layer
        )

        # MoE or Dense layer
        self.is_moe_layer = check_is_moe(config, layer_idx, is_mtp_layer)
        if self.is_moe_layer:
            self.fusion_config.PRE_MOE_FUSION = self.enable_fusion and has_tp
            self.fusion_config.POST_MOE_FUSION = self.fusion_config.PRE_MOE_FUSION and not has_pp
            self.mlp = ExaoneMoeSparseMoEBlock(
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                shared_expert_intermediate_size=config.moe_intermediate_size
                * config.num_shared_experts,
                dtype=config.torch_dtype,
                model_config=model_config,
                override_quant_config=quant_config,
                aux_stream_dict=aux_stream_dict,
                layer_idx=layer_idx,
            )
        else:
            block_size = 1
            if quant_config.quant_algo is None and quant_config.group_size is not None:
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
                # Keep sharding consistent with computed mlp_tp_size.
                # In attention-DP, mlp_tp_size==1 -> disable TP sharding here.
                overridden_tp_size=self.mlp_tp_size,
                layer_idx=layer_idx,
                reduce_output=has_mlp_tp,
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
        self.next_layer_layernorm: RMSNorm = None

    def _get_decoder_layer_quant_config(
        self, model_config: ModelConfig[ExaoneMoEConfig], layer_idx: int
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
        """Adopted from DeepseekV3DecoderLayer._compute_mlp_tp_size."""
        assert intermediate_size % block_size == 0, (
            f"intermediate_size {intermediate_size} must be divisible by block_size {block_size}."
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
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # LN has neem already applied at the previous layer except the first layer.
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )

        if self.is_moe_layer:
            hidden_states, residual = self.forward_moe(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
            )
        else:
            hidden_states, residual = self.forward_mlp(
                hidden_states=hidden_states,
                residual=residual,
            )

        return hidden_states, residual

    def forward_moe(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        def _run_moe(hidden_states, hidden_states_fp4, do_finalize):
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
                self.fusion_config.POST_MOE_FUSION
                and hidden_states.shape[0] <= self.moe_allreduce.max_token
                and self.model_config.moe_backend == "TRTLLM"
                and self.mlp.experts.has_nvfp4
                and self.is_p2p_supported
            )
        )

        hidden_states = _run_moe(hidden_states, hidden_states_fp4=None, do_finalize=do_finalize)

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
        elif self.next_layer_layernorm is not None:
            hidden_states, residual = self.next_layer_layernorm(hidden_states, residual)

        return hidden_states, residual

    def forward_mlp(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
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
        elif self.next_layer_layernorm is not None:
            hidden_states, residual = self.next_layer_layernorm(hidden_states, residual)

        return hidden_states, residual


class ExaoneMoeMTPHead(DeepseekV3MTPHead):
    """The shared head of the MTP Layer."""


class ExaoneMoeMTP(ExaoneMoeDecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[ExaoneMoEConfig],
        layer_idx: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
    ):
        super().__init__(model_config, aux_stream_dict, layer_idx)
        self.model_config = model_config
        self.mapping = model_config.mapping
        config = model_config.pretrained_config
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

        # Normalization for input embedding
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

        self.shared_head = ExaoneMoeMTPHead(model_config=model_config)

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
        def norm_embeds():
            return self.enorm(embed_tokens(input_ids))

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

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(
            hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            final_all_reduce_params=AllReduceParams(
                enable_allreduce=tp_size > 1 and not (self.model_config.mapping.enable_attention_dp)
            ),
        )

        hidden_states, _ = self.shared_head.norm(hidden_states, residual)

        return hidden_states


class ExaoneMoeModel(DecoderModel):
    def __init__(self, model_config: ModelConfig[ExaoneMoEConfig]):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=config.torch_dtype,
            mapping=model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )

        aux_stream_list = [torch.cuda.Stream() for _ in range(3)]
        self.aux_stream_dict = {
            AuxStreamType.Attention: aux_stream_list[0],
            AuxStreamType.MoeShared: aux_stream_list[0],
            AuxStreamType.MoeChunkingOverlap: aux_stream_list[1],
            AuxStreamType.MoeBalancer: aux_stream_list[2],
        }

        self.layers = nn.ModuleList(
            [
                ExaoneMoeDecoderLayer(
                    model_config=model_config,
                    aux_stream_dict=self.aux_stream_dict,
                    layer_idx=layer_idx,
                )
                for layer_idx in range(self.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        lora_params=None,
        **kwargs,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at "
                "the same time, and must specify either one."
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds.to(self.dtype)
        residual = None

        for decoder_layer in self.layers[: self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                lora_params=lora_params,
            )
        # The last LN already has been applied as a part of fusion.
        return hidden_states


@register_auto_model("ExaoneMoEForCausalLM")
class ExaoneMoeForCausalLM(SpecDecOneEngineForCausalLM[ExaoneMoeModel, ExaoneMoEConfig]):
    def __init__(
        self,
        model_config: ModelConfig[ExaoneMoEConfig],
    ):
        if (
            model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
        ):
            # NOTE: K-EXAONE does not contain the 'num_nextn_predict_layers' field,
            # which should be equal to 1. Manually set the value here if not present.
            if not hasattr(model_config.pretrained_config, "num_nextn_predict_layers"):
                model_config.pretrained_config.num_nextn_predict_layers = 1

        super().__init__(
            model=ExaoneMoeModel(model_config),
            model_config=model_config,
        )

        if (
            model_config.spec_config is not None
            and model_config.spec_config.spec_dec_mode.is_mtp_one_model()
        ):
            model_nextn = model_config.spec_config.num_nextn_predict_layers
            ckpt_nextn = self.config.num_nextn_predict_layers
            self.num_hidden_layers = self.config.num_hidden_layers
            if ckpt_nextn == 0:
                raise ValueError(
                    "No MTP module is in given checkpoint. Please check if the checkpoint supports the MTP layer "
                    "(`num_nextn_predict_layers`)."
                )
            if ckpt_nextn > 1 or model_config.spec_config.use_mtp_vanilla:
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

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional[ExaoneMoeWeightMapper] = None,  # noqa: F821
        skip_modules: Optional[List[str]] = None,
        allow_partial_loading: bool = False,
    ):
        assert isinstance(weight_mapper, ExaoneMoeWeightMapper)

        if self.draft_model is not None:
            weight_mapper.preprocess_weights(weights)

        # Weight renaming MoE is handled in ExaoneMoeWeightMapper.rename_by_params_map
        super().load_weights(
            weights=weights,
            weight_mapper=weight_mapper,
            params_map=weight_mapper.params_map,
            allow_partial_loading=allow_partial_loading,
        )

    def post_load_weights(self):
        # For the cross-layer residual+LN fusion.
        for idx, layer in enumerate(self.model.layers[: self.config.num_hidden_layers]):
            if idx == self.config.num_hidden_layers - 1:
                layer.next_layer_layernorm = self.model.norm
            else:
                layer.next_layer_layernorm = self.model.layers[idx + 1].input_layernorm
