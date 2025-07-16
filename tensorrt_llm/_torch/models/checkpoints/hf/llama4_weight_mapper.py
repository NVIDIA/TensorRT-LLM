from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Llama4ForConditionalGeneration")
class Llama4HfWeightMapper(HfWeightMapper):
    """
    Weight mapper for Llama4ForConditionalGeneration that handles the
    'language_model.' prefix removal from weight keys.
    """

    def filter_weights(self, prefix: str, weights: dict) -> dict:
        transformed_weights = {}
        for key, value in weights.items():
            if key.startswith("language_model."):
                new_key = key[len("language_model."):]
                transformed_weights[new_key] = value
            else:
                transformed_weights[key] = value

        return super().filter_weights(prefix, transformed_weights)
