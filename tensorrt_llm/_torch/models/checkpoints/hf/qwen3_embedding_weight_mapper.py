from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Qwen3ForEmbedding")
class Qwen3EmbeddingHfWeightMapper(HfWeightMapper):
    """Weight mapper for Qwen3-Embedding HF checkpoints.

    Qwen3-Embedding HF checkpoints store backbone weights without the
    ``model.`` prefix (e.g. ``layers.0.*`` instead of ``model.layers.0.*``)
    and omit ``lm_head.weight``.  FP8 quantized checkpoints produced by
    Model Optimizer already include the prefix.  Both layouts are handled
    transparently by falling back to the unprefixed key when the prefixed
    lookup returns nothing.
    """

    def filter_weights(self, prefix: str, weights: dict) -> dict:
        result = super().filter_weights(prefix, weights)
        if not result and prefix.startswith("model."):
            result = super().filter_weights(prefix[len("model.") :], weights)
        return result
