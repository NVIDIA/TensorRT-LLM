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
