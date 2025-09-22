from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Qwen2VLForConditionalGeneration")
class Qwen2VLHfWeightMapper(HfWeightMapper):
    """
    Weight mapper for Qwen2VLForConditionalGeneration that handles the
    'language_model.' prefix removal from weight keys.
    """

    def filter_weights(self, prefix: str, weights: dict) -> dict:
        transformed_weights = {}
        language_model_prefix = "model.language_model."
        for key, value in weights.items():
            if key.startswith(language_model_prefix):
                new_key = "model." + key[len(language_model_prefix):]
                transformed_weights[new_key] = value
            else:
                transformed_weights[key] = value
        return super().filter_weights(prefix, transformed_weights)
