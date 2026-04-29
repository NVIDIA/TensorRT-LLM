from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Gemma2ForCausalLM")
@register_mapper("HF", "PaliGemmaForConditionalGeneration")
class Gemma2HfWeightMapper(HfWeightMapper):
    """Weight mapper for Gemma2 and PaliGemma2.

    Gemma2 uses a "1P" (one-plus) RMSNorm convention: stored weight is (scale - 1),
    so we add +1.0 when loading. This matches the Gemma3 convention.
    """

    def should_skip_module(self, module_name: str) -> bool:
        if self.model.config.tie_word_embeddings and module_name.startswith(
                "lm_head"):
            return True

        if hasattr(self.model, "model") and hasattr(
                self.model.model, "has_custom_embed_tokens"
        ) and self.model.model.has_custom_embed_tokens and module_name == "model.embed_tokens":
            return True
        if hasattr(self.model, "has_custom_lm_head"
                   ) and self.model.has_custom_lm_head and module_name == "lm_head":
            return True

        return any(skip_module in module_name
                   for skip_module in self._skip_modules)

    def handle_manual_copy(self,
                           module_name: str,
                           module_weights: dict,
                           n: str,
                           p: nn.Parameter,
                           allow_partial_loading: bool = False) -> None:
        if "norm" in module_name:
            if not allow_partial_loading:
                assert n in module_weights
            if n in module_weights:
                p.data.copy_(module_weights[n][:] + 1)
        else:
            super().handle_manual_copy(
                module_name,
                module_weights,
                n,
                p,
                allow_partial_loading=allow_partial_loading)
