def is_nemotron_hybrid(config):
    if hasattr(config, "hybrid_override_pattern"):
        return True
    return False


def is_mla(config):
    if getattr(config, "kv_lora_rank", None) and getattr(
            config, "qk_rope_head_dim", None):
        return True
    return False
