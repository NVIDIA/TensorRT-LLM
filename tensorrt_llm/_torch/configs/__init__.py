from tensorrt_llm._torch.configs.deepseek_v3 import DeepseekV3Config


def _register_custom_configs_with_transformers() -> None:
    # Make AutoConfig.from_pretrained / AutoTokenizer.from_pretrained accept
    # model_types that TRT-LLM understands but upstream transformers does not
    # (DeepSeek-V3.2 and Kimi K2 both ship config.json with these model_types
    # and rely on TRT-LLM's local DeepseekV3Config workaround).
    #
    # Without this, transformers 5.5.x falls back to a bare PreTrainedConfig
    # that lacks attributes like `max_position_embeddings`, and
    # AutoTokenizer.from_pretrained then raises AttributeError before any
    # tokenizer can be constructed. Bypass AutoConfig.register's model_type
    # consistency check (DeepseekV3Config.model_type is "deepseek_v3") by
    # writing into the underlying mapping directly.
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    for model_type in ("deepseek_v32", "kimi_k2"):
        if model_type in CONFIG_MAPPING:
            continue
        CONFIG_MAPPING.register(model_type, DeepseekV3Config, exist_ok=True)


_register_custom_configs_with_transformers()
del _register_custom_configs_with_transformers

__all__ = ["DeepseekV3Config"]
