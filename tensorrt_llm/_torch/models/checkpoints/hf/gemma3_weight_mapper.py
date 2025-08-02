from torch import nn

from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm._torch.models.modeling_utils import register_mapper


@register_mapper("HF", "Gemma3ForCausalLM")
@register_mapper("HF", "Gemma3ForConditionalGeneration")
class Gemma3HfWeightMapper(HfWeightMapper):

    def should_skip_module(self, module_name: str) -> bool:
        if self.model.config.tie_word_embeddings and module_name.startswith(
                "lm_head"):
            return True

        # Skip loading weights for embedding and lm_head if LoRA is enabled and has custom values
        if hasattr(self.model, "model") and hasattr(
                self.model.model, 'has_custom_embed_tokens'
        ) and self.model.model.has_custom_embed_tokens and module_name == "model.embed_tokens":
            return True
        if hasattr(
                self.model, 'has_custom_lm_head'
        ) and self.model.has_custom_lm_head and module_name == "lm_head":
            return True

        return any(skip_module in module_name
                   for skip_module in self._skip_modules)

    def handle_manual_copy(self, module_name: str, module_weights: dict, n: str,
                           p: nn.Parameter) -> None:
        if 'norm' in module_name:
            p.data.copy_(module_weights[n][:] + 1)
        else:
            super().handle_manual_copy(module_name, module_weights, n, p)
