from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Qwen3VLForConditionalGeneration")
class Qwen3VLHfWeightMapper(HfWeightMapper):
    def preprocess_weights(self, weights: dict) -> dict:
        return weights

    @property
    def _head_dim(self) -> int:
        config = self.model.config
        if (head_dim := getattr(config, "head_dim", None)) is not None:
            return head_dim
        if isinstance(config, Qwen3VLTextConfig):
            num_heads = config.num_attention_heads
        elif isinstance(config, Qwen3VLVisionConfig):
            num_heads = config.num_heads
        else:
            raise TypeError(f"Unexpected config class {type(config).__name__}.")

        return config.hidden_size // num_heads
