from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Qwen2VLForConditionalGeneration")
class Qwen2VLHfWeightMapper(HfWeightMapper):
    """
    Weight mapper for Qwen2VLForConditionalGeneration that handles the
    'language_model.' prefix removal from weight keys.
    """

    def preprocess_weights(self, weights: dict) -> dict:
        """
        Preprocess weights to remove the 'model.language_model.' and 'model.visual.' prefixes.
        """
        transformed_weights = {}
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                new_key = "model." + key[len("model.language_model."):]
                transformed_weights[new_key] = value
            elif key.startswith("model.visual."):
                new_key = "visual." + key[len("model.visual."):]
                transformed_weights[new_key] = value
            else:
                transformed_weights[key] = value
        return transformed_weights
