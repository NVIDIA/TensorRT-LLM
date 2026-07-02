from tensorrt_llm._torch.configs.deepseek_v3 import DeepseekV3Config
from tensorrt_llm._torch.configs.deepseekv4 import DeepseekV4Config
from tensorrt_llm._torch.configs.gemma4_unified import (
    Gemma4UnifiedAudioConfig,
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedVisionConfig,
)
from tensorrt_llm._torch.configs.laguna import LagunaConfig


def _register_custom_configs_with_transformers() -> None:
    # Make AutoConfig.from_pretrained / AutoTokenizer.from_pretrained accept
    # model_types that TRT-LLM understands but upstream transformers does not
    # (DeepSeek-V3.2, Kimi K2, and Laguna ship config.json with these
    # model_types and rely on TRT-LLM's local config workarounds; likewise
    # the gemma4_unified family).
    #
    # Without this, transformers 5.5.x falls back to a bare PreTrainedConfig
    # that lacks attributes like `max_position_embeddings`, and
    # AutoTokenizer.from_pretrained then raises AttributeError before any
    # tokenizer can be constructed. Bypass AutoConfig.register's model_type
    # consistency check for aliases (for example, DeepseekV3Config.model_type
    # is "deepseek_v3") by writing into the underlying mapping directly.
    # Registration only fills gaps: when the installed transformers already
    # ships a model_type, the native class wins.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    custom_configs = {
        "deepseek_v32": DeepseekV3Config,
        "kimi_k2": DeepseekV3Config,
        "deepseek_v4": DeepseekV4Config,
        "laguna": LagunaConfig,
        "gemma4_unified": Gemma4UnifiedConfig,
        "gemma4_unified_text": Gemma4UnifiedTextConfig,
        "gemma4_unified_vision": Gemma4UnifiedVisionConfig,
        "gemma4_unified_audio": Gemma4UnifiedAudioConfig,
    }
    for model_type, config_class in custom_configs.items():
        if model_type in CONFIG_MAPPING:
            continue
        CONFIG_MAPPING.register(model_type, config_class, exist_ok=True)


_register_custom_configs_with_transformers()
del _register_custom_configs_with_transformers

__all__ = [
    "DeepseekV3Config",
    "DeepseekV4Config",
    "Gemma4UnifiedAudioConfig",
    "Gemma4UnifiedConfig",
    "Gemma4UnifiedTextConfig",
    "Gemma4UnifiedVisionConfig",
    "LagunaConfig",
]
