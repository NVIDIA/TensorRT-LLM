from .rocket import RocketKVCacheManager, RocketVanillaAttention


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}"
        )


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    raise ValueError(
        f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
    )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )
