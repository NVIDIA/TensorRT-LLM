import importlib
import logging
import os

_logger = logging.getLogger(__name__)

# Import each custom model individually so that models with transitive TRT-LLM
# dependencies (e.g., NemotronH needing mamba layernorm_gated) don't prevent
# other models from loading in standalone mode.
_MODEL_MODULES = {
    "modeling_deepseek": ["DeepSeekV3ForCausalLM"],
    "modeling_gemma3n": ["Gemma3nForCausalLM", "Gemma3nForConditionalGeneration"],
    "modeling_gemma4": ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"],
    "modeling_glm4_moe_lite": ["Glm4MoeLiteForCausalLM"],
    "modeling_kimi_k2": ["KimiK2ForCausalLM", "KimiK25ForConditionalGeneration"],
    "modeling_minimax_m2": ["MiniMaxM2ForCausalLM"],
    "modeling_mistral3": ["Mistral3ForConditionalGenerationAD", "Mistral4ForCausalLM"],
    "modeling_nemotron_flash": ["NemotronFlashForCausalLM", "NemotronFlashPreTrainedTokenizerFast"],
    "modeling_nemotron_h": ["NemotronHForCausalLM"],
    "modeling_qwen3_5_moe": ["Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"],
}

if os.environ.get("AD_USE_IR_MODELS"):
    _MODEL_MODULES["modeling_deepseek_ir"] = ["DeepSeekV3ForCausalLM"]
    _MODEL_MODULES["modeling_nemotron_h_ir"] = ["NemotronHForCausalLM"]
    _MODEL_MODULES["modeling_qwen3_5_moe_ir"] = [
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ]

__all__ = []
for _module_name, _names in _MODEL_MODULES.items():
    try:
        _mod = importlib.import_module(f".{_module_name}", __name__)
        for _name in _names:
            globals()[_name] = getattr(_mod, _name)
            if _name not in __all__:
                __all__.append(_name)
    except (ImportError, ModuleNotFoundError, ValueError) as _exc:
        _logger.debug("Skipping custom model %s: %s", _module_name, _exc)

__all__ = tuple(__all__)
