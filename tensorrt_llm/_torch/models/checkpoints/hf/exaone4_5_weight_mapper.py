from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Exaone4_5_ForConditionalGeneration")
class Exaone4_5HfWeightMapper(HfWeightMapper):
    def __init__(self):
        super().__init__()

    def preprocess_weights(self, weights: dict):
        """Rename HF checkpoint prefixes; supports plain dict and ConsumableWeightsDict."""
        is_consumable = isinstance(weights, ConsumableWeightsDict)
        renamed = {}
        for key, value in weights.items():
            if key.startswith("model.visual."):
                new_key = key.replace("model.visual.", "visual.")
                renamed[new_key] = value
            elif key.startswith("model.language_model."):
                new_key = key.replace("model.language_model.", "model.")
                renamed[new_key] = value
            else:
                renamed[key] = value
        if is_consumable:
            return ConsumableWeightsDict(renamed)
        return renamed
