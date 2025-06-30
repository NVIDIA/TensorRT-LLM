def is_nemotron_hybrid(config):
    if hasattr(config, "hybrid_override_pattern"):
        return True
    return False


def is_mla(config):
    if (hasattr(config, "kv_lora_rank") and config.kv_lora_rank is not None
            and hasattr(config, "qk_rope_head_dim")
            and config.qk_rope_head_dim is not None):
        return True
    return False
