import transformers


def is_nemotron_hybrid(config):
    if hasattr(config, "hybrid_override_pattern"
               ) and config.hybrid_override_pattern is not None and len(
                   config.hybrid_override_pattern) > 0:
        return True
    return False


def is_mla(config):
    if getattr(config, "kv_lora_rank", None) and getattr(
            config, "qk_rope_head_dim", None):
        return True
    return False


def is_qwen3_next(config):
    return hasattr(
        config, 'architectures'
    ) and config.architectures is not None and config.architectures[
        0] == 'Qwen3NextForCausalLM'


# TODO: remove this once the transformers can support all of those models in _CONFIG_REGISTRY
class LazyConfigDict(dict):

    def __getitem__(self, key):
        import tensorrt_llm._torch.configs as configs
        return getattr(configs, super().__getitem__(key))


_CONFIG_REGISTRY: dict[str, type[transformers.PretrainedConfig]] = LazyConfigDict(
    deepseek_v32="DeepseekV3Config",
)  # NOTE: HF config.json uses deepseek_v32 as model_type but with same DSV3 config class


def load_pretrained_config(model_name_or_path: str,
                           trust_remote_code: bool = False,
                           **kwargs) -> transformers.PretrainedConfig:
    config_dict, _ = transformers.PretrainedConfig.get_config_dict(
        model_name_or_path, **kwargs)
    model_type = config_dict.get("model_type")
    if model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[model_type]
        model_config = config_class.from_pretrained(model_name_or_path,
                                                    **kwargs)
    else:
        model_config = transformers.AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code)
    return model_config
