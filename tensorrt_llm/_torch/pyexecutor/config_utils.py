def is_nemotron_hybrid(config):
    return getattr(config, "hybrid_override_pattern", None) is not None \
        or (isinstance(config, dict) and config.get("hybrid_override_pattern", None) is not None)


def is_mla(config):
    if hasattr(config, "kv_lora_rank"):
        assert hasattr(
            config, "qk_rope_head_dim"
        ), "both of kv_lora_rank and qk_rope_head_dim are required."
        return True
    return False
