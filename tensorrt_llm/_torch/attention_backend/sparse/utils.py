from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

from .aether import AetherVanillaAttention
from .dsa import DSACacheManager, DSATrtllmAttention
from .rocket import (RocketKVCacheManager, RocketTrtllmAttention,
                     RocketVanillaAttention)


def get_sparse_attn_kv_cache_manager(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attn_config.algorithm == "dsa":
        return DSACacheManager
    elif sparse_attn_config.algorithm == "skip_softmax":
        return KVCacheManager
    elif sparse_attn_config.algorithm == "aether":
        # AETHER uses standard KV cache with block metadata overlay
        return KVCacheManager
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}"
        )


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    elif sparse_attn_config.algorithm == "aether":
        return AetherVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attn_config.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_attn_config.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_attn_config.algorithm == "skip_softmax":
        return TrtllmAttention
    elif sparse_attn_config.algorithm == "aether":
        # AETHER currently routes through VanillaAttention backend
        # Future: Create optimized AetherTrtllmAttention
        return AetherVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "aether":
        # AETHER can work with FlashInfer via VanillaAttention fallback
        return AetherVanillaAttention
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )

