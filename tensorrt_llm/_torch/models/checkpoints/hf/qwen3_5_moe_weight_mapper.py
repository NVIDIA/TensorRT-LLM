from torch import nn

from tensorrt_llm.logger import logger
from tensorrt_llm._torch.models.checkpoints.hf.qwen3_5_weight_mapper import Qwen3_5HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper
from tensorrt_llm._torch.modules.fused_moe.interface import MoE, MoEWeightLoadingMode


# Register Qwen3_5Moe configs, TODO: Remove this once we have a proper transformers package
from transformers import AutoConfig, PretrainedConfig  # isort: skip

class Qwen3_5MoeTextConfig(PretrainedConfig):
    model_type = "qwen3_5_moe_text"
    base_config_key = "text_config"
    
    def __init__(
        self,
        tie_word_embeddings=False,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

class Qwen3_5MoeVisionConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    base_config_key = "vision_config"

class Qwen3_5MoeConfig(PretrainedConfig):
    model_type = "qwen3_5_moe"
    sub_configs = {"vision_config": Qwen3_5MoeVisionConfig, "text_config": Qwen3_5MoeTextConfig}
    
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()

        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)

logger.warning_once(
    "transformers version below 5.1.0 does not support 'Qwen3_5MoeConfig'. "
    "Register Qwen3_5MoeConfig to mimic the Qwen3_5Moe model.",
    key="QWEN3_5_MOE_REGISTER_WARNING"
)
AutoConfig.register(Qwen3_5MoeTextConfig.model_type, Qwen3_5MoeTextConfig)
AutoConfig.register(Qwen3_5MoeConfig.model_type, Qwen3_5MoeConfig)
# End of the config register.


@register_mapper("HF", "Qwen3_5MoeForConditionalGeneration")
class Qwen3_5MoeHfWeightMapper(Qwen3_5HfWeightMapper):
    def is_special_instance_module(self, module: nn.Module) -> bool:
        return isinstance(module, MoE)
    
    def handle_special_instance_module(
        self,
        module: nn.Module,
        module_name: str,
        module_weights: dict,
        allow_partial_loading: bool = False,
    ) -> None:
        if isinstance(module, MoE):
            # NOTE: to check the weight loading mode, if gate_up_proj in HF ckpt, then FUSED_GATE_UP_PROJ
            weight_loading_mode = MoEWeightLoadingMode.VANILLA
            for weight_name, weight_value in module_weights.items():
                if "gate_up_proj" in weight_name:
                    weight_loading_mode = MoEWeightLoadingMode.FUSED_GATE_UP_PROJ
                    break
            module.weight_loading_mode = weight_loading_mode
            
            updated_module_weights = {}
            for weight_name, weight_value in module_weights.items():
                if weight_loading_mode == MoEWeightLoadingMode.VANILLA:
                    new_weight_name = weight_name.replace(
                        "gate_proj", "w1").replace("up_proj",
                                                "w3").replace("down_proj", "w2")
                    new_weight_value = weight_value
                elif weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ:
                    new_weight_name = weight_name
                    # NOTE: transpose the weights, check the FusedMoEMethodBase in fused_moe/quantization.py, 
                    #   the load_expert_weights_to_dst function, "elif weight_loading_mode == MoEWeightLoadingMode.FUSED_GATE_UP_PROJ" line
                    # Note that Qwen3VLMoE has the same weight loading mode, but the weights are not transposed, 
                    #   because the HF weights layout are different
                    new_weight_value = weight_value.transpose(-2, -1).contiguous()
                else:
                    raise NotImplementedError(
                        f"Unsupported weight loading mode in MoE: {weight_loading_mode}"
                    )
                updated_module_weights[new_weight_name] = new_weight_value
            
            module.load_weights(
                weights=[updated_module_weights], allow_partial_loading=allow_partial_loading
            )

    @property
    def _num_kv_heads(self) -> int:
        config = self._model.config
        if isinstance(config, Qwen3_5MoeTextConfig):
            num_kv_heads = getattr(config, "num_key_value_heads", None)
            if num_kv_heads is None:
                num_kv_heads = config.num_attention_heads
        elif isinstance(config, Qwen3_5MoeVisionConfig):
            num_kv_heads = config.num_heads
        else:
            raise TypeError(
                "Expected `Qwen3_5MoeTextConfig` or `Qwen3_5MoeVisionConfig`, "
                f"got {type(config).__name__}"
            )

        return num_kv_heads
