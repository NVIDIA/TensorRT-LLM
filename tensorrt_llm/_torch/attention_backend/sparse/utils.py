from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

from .dsa import DSACacheManager, DSATrtllmAttention
from .minimax_m3 import (get_minimax_m3_attention_backend_cls,
                         get_minimax_m3_kv_cache_manager_cls)
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
    elif sparse_attn_config.algorithm == "minimax_m3":
        return get_minimax_m3_kv_cache_manager_cls()
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attn_config.algorithm}"
        )


def get_vanilla_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "rocket":
        return RocketVanillaAttention
    elif sparse_attn_config.algorithm == "minimax_m3":
        return get_minimax_m3_attention_backend_cls()
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
    elif sparse_attn_config.algorithm == "minimax_m3":
        # The MiniMax-M3 sparse algorithm runs in Python through the
        # model-layer override; this backend exists so the standard
        # `create_attention(...)` dispatch in `Attention.__init__`
        # returns an instantiable AttentionBackend under the trtllm
        # attention backend slot.
        return get_minimax_m3_attention_backend_cls()
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attn_config.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attn_config: "SparseAttentionConfig"):
    if sparse_attn_config.algorithm == "minimax_m3":
        return get_minimax_m3_attention_backend_cls()
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attn_config.algorithm}"
    )
