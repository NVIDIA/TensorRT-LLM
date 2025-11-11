from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Qwen3VLForConditionalGeneration")
class Qwen3VLHfWeightMapper(HfWeightMapper):
    def preprocess_weights(self, weights: dict) -> dict:
        return weights
