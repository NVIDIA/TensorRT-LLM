import os
import re
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch.modules.qk_norm_attention import QKNormRoPEAttention
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.quantization import QuantAlgo

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import (
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
    RopeParams,
)
from ..distributed import AllReduce, AllReduceParams, MoEAllReduce
from ..model_config import ModelConfig
from ..models.modeling_deepseekv3 import Deepseekv3MoE
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.gated_mlp import GatedMLP
from ..modules.linear import TensorParallelMode
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import AuxStreamType
from .modeling_utils import (
    DecoderModel,
    DecoderModelForCausalLM,
    EagerFusionConfig,
    register_auto_model,
)

# fmt: off
# TODO: Remove this once we have a proper transformers package
from transformers import AutoConfig, PretrainedConfig  # isort: skip

class ExaoneMoEConfig(PretrainedConfig):
    model_type = "exaone_moe"

print(
    "transformers does not support 'ExaoneMoEConfig'. "
    "Register ExaoneMoEConfig to mimic the ExaoneMoE model."
)
AutoConfig.register(ExaoneMoEConfig.model_type, ExaoneMoEConfig)
# End of the config register.
# fmt: on


PRINT_DEBUG = False


def debug_print(tag: str, x: torch.Tensor, layer_idx: Optional[int] = None):
    if PRINT_DEBUG and not torch.cuda.is_current_stream_capturing():
        tag = tag if layer_idx is None else f"layer.{layer_idx}.{tag}"
        x = x.float()
        print(
            f"[DEBUG_PRINT] TR | {tag:20s} | l1_norm {x.abs().mean():10.4f} | mean {x.mean():10.4f}"
        )


def check_is_sliding(config: ExaoneMoEConfig, layer_idx: int) -> bool:
    """
    Check if the current layer is a sliding window (local attention) layer.
    """
    if config.sliding_window is None:
        return False
    if isinstance(config.sliding_window_pattern, int):
        return ((layer_idx + 1) % config.sliding_window_pattern) != 0
    elif isinstance(config.sliding_window_pattern, str):
        assert isinstance(config.sliding_window, int), (
            f"Sliding window must be positive integer, but got {config.sliding_window}"
        )
        return (
            layer_idx != config.num_hidden_layers - 1
            and config.sliding_window_pattern[layer_idx % len(config.sliding_window_pattern)] == "L"
        )
    return False


def check_is_moe(config: ExaoneMoEConfig, layer_idx: int) -> bool:
    """
    Check if the current layer is a MoE layer.
    """
    return hasattr(config, "is_moe_layer") and config.is_moe_layer[layer_idx]


class ExaoneMoeAttention(QKNormRoPEAttention):
    def __init__(
        self,
        model_config: ModelConfig[ExaoneMoEConfig],
        layer_idx: Optional[int] = None,
        fuse_qk_norm_rope: bool = False,
        disable_deep_gemm: bool = False,
    ):
        config = model_config.pretrained_config

        self.attention_window_size = None

        # NOTE: In ExaoneMoe, only sliding layers apply rope.
        self.sliding_window = config.sliding_window
        self.is_sliding = check_is_sliding(config, layer_idx)
        pos_embd_params = None
        if self.sliding_window is None or self.is_sliding:
            self.attention_window_size = config.sliding_window

            pos_embd_params = PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gpt_neox,
                rope=RopeParams.from_config(config),
            )

        fuse_qk_norm_rope = self.is_sliding and fuse_qk_norm_rope

        # TODO: Fusing qk norm with rope has an issue that slightly hurts accuracy.
        assert fuse_qk_norm_rope is False, "Fusing qk norm and rope is having issue now"

        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=pos_embd_params,
            fuse_qk_norm_rope=fuse_qk_norm_rope,
            skip_rope=self.sliding_window and not self.is_sliding,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            disable_deep_gemm=disable_deep_gemm,
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
        # TODO LoRA has not been tested yet but there is no need to prevent it.
        assert lora_params is None, "LORA is not supported for ExaoneMoeAttention"

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
        config = model_config.pretrained_config
        self.layer_idx = layer_idx
        self.is_quanted = (
            model_config.quant_config and model_config.quant_config.quant_mode.has_any_quant()
        )

        self.mapping = model_config.mapping
        mapping = self.mapping
        self.enable_attention_dp = mapping.enable_attention_dp
        self.mlp_tp_size = mapping.tp_size

        from tensorrt_llm._ipc_utils import can_access_peer

        self.is_p2p_supported = can_access_peer(mapping)

        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
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

        # TODO(jaedeokk): Finalize parallelism and fusion configuration.
        # has_tp = mapping.has_tp()

        self.disable_deep_gemm = False
        quant_config = getattr(model_config, "quant_config", None)
        if quant_config is not None:
            # TODO(jaedeokk): Check if
            # ExaoneMoe fp8 has an illegal memory access issue with deep_gemm.
            self.disable_deep_gemm = (
                getattr(quant_config, "quant_algo", None) == QuantAlgo.FP8_BLOCK_SCALES
            )

        self.fusion_config = EagerFusionConfig()
        self.enable_fusion = os.environ.get("TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED", "0") == "0"
        self.enable_fusion &= not self.enable_attention_dp

        self.disable_attn_allreduce = (
            self.fusion_config.PRE_MOE_FUSION
            or self.fusion_config.PRE_MLP_FUSION
            or self.mapping.tp_size == 1
            or self.enable_attention_dp
        )

        self.self_attn = ExaoneMoeAttention(
            model_config,
            layer_idx=layer_idx,
            disable_deep_gemm=self.disable_deep_gemm,
        )

        # MoE or Dense layer
        if check_is_moe(config, layer_idx):
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
            self.mlp = GatedMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                bias=False,
                dtype=config.torch_dtype,
                config=model_config,
                layer_idx=layer_idx,
                disable_deep_gemm=self.disable_deep_gemm,
            )

        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=config.torch_dtype
        )
        self.mapping = model_config.mapping

    def forward(
        self,
        position_ids: torch.LongTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        debug_print("ln_out", hidden_states, self.layer_idx)

        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not (self.disable_attn_allreduce)),
            **kwargs,
        )

        debug_print("attn_out", hidden_states, self.layer_idx)

        # residual = hidden_states
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual=residual)

        debug_print("post_ln_out", hidden_states, self.layer_idx)
        debug_print("post_ln_res", residual, self.layer_idx)

        hidden_states = self.mlp(hidden_states)
        debug_print("mlp_out", hidden_states, self.layer_idx)

        hidden_states = hidden_states + residual
        debug_print("layer_out", hidden_states, self.layer_idx)

        return hidden_states, residual


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
        spec_metadata: Optional[SpecMetadata] = None,
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

        debug_print("inputs_embeds", inputs_embeds)
        residual = None

        for decoder_layer in self.layers[: self.num_hidden_layers]:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                lora_params=lora_params,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


@register_auto_model("ExaoneMoEForCausalLM")
class ExaoneMoeForCausalLM(DecoderModelForCausalLM[ExaoneMoeModel, ExaoneMoEConfig]):
    def __init__(
        self,
        model_config: ModelConfig[ExaoneMoEConfig],
    ):
        model_config.pretrained_config.torch_dtype = torch.bfloat16
        super().__init__(
            ExaoneMoeModel(model_config),
            config=model_config,
            hidden_size=model_config.pretrained_config.hidden_size,
            vocab_size=model_config.pretrained_config.vocab_size,
        )

    def load_weights(
        self,
        weights: Dict,
        weight_mapper: Optional["BaseWeightMapper"] = None,  # noqa: F821
        skip_modules: Optional[List[str]] = None,
        allow_partial_loading: bool = False,
    ):
        # MoE namining pattern.
        moe_weight_patterns = {
            "gate_proj": "w1",
            "up_proj": "w3",
            "down_proj": "w2",
        }

        module_names = list(weights)
        for name in module_names:
            if "mlp.e_score_correction_bias" in name:
                # Move bias into the gate module.
                new_name = name.replace(
                    "mlp.e_score_correction_bias", "mlp.gate.e_score_correction_bias"
                )
            else:
                # MoE Weight Remapping.
                new_name = name
                for k, v in moe_weight_patterns.items():
                    pattern = rf"(experts\.\d+\.){k}\b"
                    new_name = re.sub(pattern, rf"\1{v}", new_name)

            # Remap the name-parameter pair if needed.
            if new_name != name:
                weights[new_name] = weights.pop(name)

        skip_modules = skip_modules or []
        super().load_weights(weights, weight_mapper, skip_modules, allow_partial_loading)
